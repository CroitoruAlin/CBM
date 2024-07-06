import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from einops.layers.torch import Rearrange
import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

        self.planes = planes

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes,no_patches):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]
        self.masking = MaskIn(no_patches=no_patches)
        self.conv1_initial = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1_final = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)


    def forward(self, x, percentage, probabilities=None):        
        x = self.masking(x, percentage, probabilities)
        out = self.conv1_initial(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1_final(out))
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