import copy
import torchvision
from torch import optim
import torch
import numpy as np
import arguments
from data_handlers import get_train_val_loaders_cifar, get_test_loader_cifar, get_train_loader_tiny_imagenet, \
    get_valid_loader_tiny_imagenet, get_train_val_loaders_food101, get_test_loader_food101

from models.resnet import ResNet18
from models.wide_resnet import Wide_ResNet
from resnet_train import Trainer
from fibonacci import fibonacci
import math
def build_optimizer_resnet(model):
    return optim.SGD(model.parameters(), lr=0.1,
                               weight_decay=5e-4,
                               momentum=0.9)

def build_optimizer_resnet_tin(model):
    return optim.SGD(model.parameters(), lr=0.1,
                               weight_decay=1e-4,
                               momentum=0.9)

def baseline(num_epochs):
    return [0] * num_epochs

def lin_repeat():
    v = fibonacci(length=7)
    for i in range(1,len(v)):
        v[i]=math.log(v[i])/(math.log(v[6])/0.4036067977500615)
    v=v[2:]
    v[0] = 0.07
    return v

def lin_repeat_tin():
    v = fibonacci(length=7)
    for i in range(1,len(v)):
        v[i]=math.log(v[i])/(math.log(v[6])/0.6036067977500615)
    v=v[2:]
    v[0] = 0.1
    return v
def train_food101(args):
    args.num_classes = 101
    args.decay_epoch = 32
    args.decay_step = 30
    args.stop_decay_epoch = 93
    args.num_epochs = 200
    args.model_name = 'r18_food101test224'
    curriculum = lin_repeat_tin()
    args.percent = curriculum * 40
    #args.percent = baseline(200)
    print(args.percent)
    print(len(args.percent))
    resnet18 = ResNet18(num_classes=args.num_classes)
    train_loader= get_train_val_loaders_food101(dataset=torchvision.datasets.Food101,resize=64)
    test_loader = get_test_loader_food101(dataset=torchvision.datasets.Food101,resize=64)
    args.testlo=test_loader
    trainer = Trainer(resnet18, train_loader, test_loader, args,build_optimizer_resnet)
    trainer.train()

def train_cifar10(args):
    args.num_classes = 10
    args.decay_epoch = 32
    args.decay_step = 30
    args.stop_decay_epoch = 93
    args.num_epochs = 200
    args.model_name = 'r18_cif10'
    curriculum = lin_repeat()
    args.percent = curriculum * 40
    print(args.percent)
    print(len(args.percent))
    resnet18 = ResNet18(num_classes=args.num_classes)
    train_loader, val_loader = get_train_val_loaders_cifar(val_size=64, dataset=torchvision.datasets.CIFAR10)
    test_loader = get_test_loader_cifar(dataset=torchvision.datasets.CIFAR10)
    args.testlo=test_loader
    trainer = Trainer(resnet18, train_loader, test_loader, args,build_optimizer_resnet)
    trainer.train()
def train_cifar100(args):
    args.num_classes = 100
    args.decay_epoch = 32
    args.decay_step = 30
    args.stop_decay_epoch = 93
    args.num_epochs = 200
    args.model_name = 'r18_cif100'
    curriculum = lin_repeat()
    args.percent = curriculum * 40
    print(args.percent)
    print(len(args.percent))
    resnet18 = ResNet18(num_classes=args.num_classes)
    train_loader, val_loader = get_train_val_loaders_cifar(val_size=64, dataset=torchvision.datasets.CIFAR100)
    test_loader = get_test_loader_cifar(dataset=torchvision.datasets.CIFAR100)
    args.testlo=test_loader
    trainer = Trainer(resnet18, train_loader, test_loader, args,build_optimizer_resnet)
    trainer.train()
def train_tinyimagenet(args):
    args.num_classes = 200
    args.decay_epoch = 32
    args.decay_step = 30
    args.stop_decay_epoch = 93
    args.num_epochs = 100
    args.model_name = 'r18_tin'
    curriculum = lin_repeat_tin()
    args.percent = curriculum * 20
    print(args.percent)
    print(len(args.percent))
    resnet18 = ResNet18(num_classes=args.num_classes)
    train_loader, class_to_idx = get_train_loader_tiny_imagenet(path="./data")
    val_loader = get_valid_loader_tiny_imagenet(path="./data", class_to_idx=class_to_idx)
    args.testlo=val_loader
    trainer = Trainer(resnet18, train_loader, val_loader, args,build_optimizer_resnet_tin)
    trainer.train()

