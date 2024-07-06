'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops.layers.torch import Rearrange
import cv2
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_planes,
            planes,
            stride=1,
    ):
        super(BasicBlock, self).__init__()

        self.planes = planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut_kernel = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # nn.init.normal_(m.weight, mean=0, std=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes,no_patches, kernel_size_avg_pool=4):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.masking = MaskIn(no_patches=no_patches)
        self.conv1_initial = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_initial = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.kernel_size_avg_pool=kernel_size_avg_pool
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # nn.init.normal_(m.weight, mean=0, std=0.5)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, percentage, probabilities=None):
        x = self.masking(x, percentage, probabilities)
        out = self.conv1_initial(x)
        out = F.relu(self.bn1_initial(out))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        return out


class MaskIn(nn.Module):
    def __init__(self, no_patches):
        super().__init__()
        self.unfold = Rearrange('b c (p1 w) (p2 h) -> b c (p1 p2) w h', p1=no_patches, p2=no_patches)
        self.fold = Rearrange('b c (p1 p2) w h -> b c (p1 w) (p2 h)', p1=no_patches, p2=no_patches)
        self.arrange_mask = Rearrange('b c p w h -> b p c w h')
        self.undo = Rearrange('b p c w h -> b c p w h')
        self.length_indexes = no_patches ** 2
        self.training = True

    def forward(self, x, percentage, probabilities):
        #probabilities=None
        if probabilities is None:
            probabilities = torch.ones((x.shape[0], self.length_indexes))
            probabilities /= torch.sum(probabilities,dim=1, keepdim=True)
        if self.training and percentage!=0:
            if percentage<0.07:
                percentage=0.07
            b,c,_,_= x.shape
            # permutation = torch.argsort(torch.rand((b,self.length_indexes)), dim=-1)
            num_samples = int(percentage*self.length_indexes)
            if num_samples !=0:
                mask_indexes = torch.multinomial(probabilities, num_samples=int(percentage*self.length_indexes))
            else:
                mask_indexes = torch.Tensor([])
            # mask_indexes = permutation[:, :int(permutation.shape[1] * percentage)]
            mask = torch.ones_like(x)

            x_unfold = self.unfold(x)
            mask = self.unfold(mask)
            mask = self.arrange_mask(mask)
            mask[torch.arange(0,b).unsqueeze(-1), mask_indexes] = 0.
            mask = self.undo(mask)

            x_masked = x_unfold * mask
            return self.fold(x_masked)
        else:
            return x

    def train(self: nn.Module, mode: bool = True) -> nn.Module:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self
    def eval(self: nn.Module) -> nn.Module:
        return self.train(False)
def ResNet18(num_classes, no_patches=4, kernel_size_avg_pool=4):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                  kernel_size_avg_pool=kernel_size_avg_pool, no_patches=no_patches)

