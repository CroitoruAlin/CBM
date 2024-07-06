import os,sys
import cv2
from patchify import patchify, unpatchify
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from test import test

class Trainer:
    def __init__(self, model, train_loader, val_loader,

  args,
                 build_optimizer):
        self.args = args
        self.model = model
        self.optimizer = build_optimizer(model)
        self.ce_loss = F.cross_entropy
        self.model = self.model.to(self.args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self):
        best_accuracy = 0.
        for epoch in range(1, self.args.num_epochs + 1):
            print('Epoch', epoch)

            if epoch == self.args.decay_epoch and epoch < self.args.stop_decay_epoch:
                for param in self.optimizer.param_groups:
                    param['lr'] = param['lr'] / 10
                print(f"Learning rate updated to {param['lr']}")
                self.args.decay_epoch += self.args.decay_step

            self._train_epoch(epoch-1)
            test_acc = test(self.model, self.args.device, self.val_loader)
            test(self.model, self.args.device,self.args.testlo)
            if best_accuracy < test_acc:
                best_accuracy = test_acc
                os.makedirs("./saved_models",exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join("saved_models", self.args.model_name + ".pth"))
                print('New best accuracy. Model Saved!')
    def _train_epoch(self,epoch):
        self.model.train()

        pbar = tqdm(self.train_loader)
        correct = 0.
        processed = 0.
        steps, length = 0, len(self.train_loader)
        update_points = []
        self.model = self.model.to('cuda')
        for (data, target, probability) in pbar:
            data, target, probability = data.to('cuda'), target.to('cuda'), probability.to('cuda')

            y_pred = self.model(data, self.args.percent[epoch], probability)
            loss = self.ce_loss(y_pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            steps += 1
        print(f'Loss={loss.item()} Accuracy={100 * correct / processed:0.2f}')
        return 100 * correct / processed