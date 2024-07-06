# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
from torch.utils.data import random_split
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import csv
import os
import cv2
import torch
import math
import torchvision
from torch import randperm, default_generator
from torch.utils.data import Subset, DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.datasets.folder import default_loader, pil_loader, accimage_loader
from torchvision.transforms import transforms
import pandas as pd
#
from sklearn.model_selection import train_test_split
import random
from einops.layers.torch import Rearrange
from einops import rearrange
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#from models.resnet import MaskIn
from torchvision.transforms.functional import to_pil_image
import cv2
class SeaTest():
    def __init__(self, dataset=None, transformations=None):
        self.dataset_train = dataset
        self.transformations = transformations
    def __getitem__(self, index):
        (img, label) = self.dataset_train[index]
        if self.transformations is not None:
            return self.transformations(img), label
        return img, label
    def __len__(self):
        return len(self.dataset_train)

class SeaAnimals():
    def __init__(self, dataset=None, transformations=None, should_download=True):
        self.dataset_train = dataset
        self.transformations = transformations
        self.probs=[]
        for i in range(len(self.dataset_train)):
            (imeg, laebl) = self.dataset_train[i]
            self.probs.append(self.get_probability(imeg))
    def get_probability(self, img):
        bla=cv2.CV_64F
        
        transformz = transforms.Compose([
        transforms.Resize((224,224))] #64 for non cvt
        )
        img = transformz(img)
        innp=np.asarray(img)
        dx=cv2.Sobel(innp,bla,1,0)
        dy=cv2.Sobel(innp,bla,0,1)
        mag=np.sqrt(dx**2+dy**2)
        mag=torch.Tensor(mag.swapaxes(0,-1).swapaxes(1,2))
        p=4
        image2=rearrange(mag, 'c (p1 w) (p2 h) -> (p1 p2) w h c', p1=p, p2=p)
        image2=image2.numpy()
        tempor=[]
        for kk in range(len(image2)):
            tempor.append(np.mean(image2[kk]))
        tempor=np.asarray(tempor)
        tempor=tempor/sum(tempor)
        return tempor
    def __getitem__(self, index):
        (img, label) = self.dataset_train[index]
        if self.transformations is not None:
            return self.transformations(img), label, self.probs[index]
        return img, label
    def __len__(self):
        return len(self.dataset_train)

def seaanimals_train_test(batch_size=64, resize=224):
    root = os.path.join('./data/archive(4)/')
    transform = transforms.Compose([
            transforms.Resize((resize,resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
        )

    temp_data = datasets.ImageFolder(root)
    train_indices, test_indices, _, _ = train_test_split(
      range(len(temp_data)),
      temp_data.targets,
      stratify=temp_data.targets,
      test_size=0.1,
      random_state=42
    )
    train_dataset = Subset(temp_data , train_indices)
    val_dataset = Subset(temp_data , test_indices)

    #n_data = len(temp_data) 
    #n_train = int(0.9 * n_data)
    #n_val = n_data - n_train
    #train_dataset, val_dataset =  random_split(temp_data, [n_train, n_val])


    train_data=SeaAnimals(transformations=transform, dataset=train_dataset)
    val_data = SeaTest(transformations=transform, dataset=val_dataset)
        #train_ds = Subset(train_data, indices[0: len(train_data)])
    print(f'Train data size {len(train_data)}')

    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(val_data, batch_size=batch_size)
    #train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_dataset, val_dataset

class AHE():
    def __init__(self, dataset_path='./data', transformations=None, should_download=True):
        self.root = os.path.join(dataset_path)
        self.dataset_train = datasets.ImageFolder(self.root)
        self.transformations = transformations
        self.probs=[]
        for i in range(len(self.dataset_train)):
            (imeg, laebl) = self.dataset_train[i]
            self.probs.append(self.get_probability(imeg))
    def get_probability(self, img):
        bla=cv2.CV_64F
        
        transformz = transforms.Compose([
        transforms.Resize((128,128))] #64 for non cvt
        )
        img = transformz(img)
        innp=np.asarray(img)
        dx=cv2.Sobel(innp,bla,1,0)
        dy=cv2.Sobel(innp,bla,0,1)
        mag=np.sqrt(dx**2+dy**2)
        mag=torch.Tensor(mag.swapaxes(0,-1).swapaxes(1,2))
        p=4
        image2=rearrange(mag, 'c (p1 w) (p2 h) -> (p1 p2) w h c', p1=p, p2=p)
        image2=image2.numpy()
        tempor=[]
        for kk in range(len(image2)):
            tempor.append(np.mean(image2[kk]))
        tempor=np.asarray(tempor)
        tempor=tempor/sum(tempor)
        return tempor
    def __getitem__(self, index):
        (img, label) = self.dataset_train[index]
        if self.transformations is not None:
            return self.transformations(img), label, self.probs[index]
        return img, label
    def __len__(self):
        return len(self.dataset_train)

def ahe_train(batch_size=64, resize=128):
    transform = transforms.Compose([
            transforms.Resize((resize,resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
        )
    train_data=AHE(transformations=transform, dataset_path = './data/archive(5)/train')
        #train_ds = Subset(train_data, indices[0: len(train_data)])
    print(f'Train data size {len(train_data)}')
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader
def ahe_test(batch_size=64,resize=128):
    test_changes = transforms.Compose([transforms.Resize((resize,resize)),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))])
    test_data = datasets.ImageFolder('./data/archive(5)/test', transform= test_changes)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader

class Balls():
    def __init__(self, dataset_path='./data', transformations=None, should_download=True):
        self.root = os.path.join(dataset_path)
        self.dataset_train = datasets.ImageFolder(self.root)
        self.transformations = transformations
        self.probs=[]
        for i in range(len(self.dataset_train)):
            (imeg, laebl) = self.dataset_train[i]
            self.probs.append(self.get_probability(imeg))
    def get_probability(self, img):
        bla=cv2.CV_64F
        
        transformz = transforms.Compose([
        transforms.Resize((224,224))] #64 for non cvt
        )
        img = transformz(img)
        innp=np.asarray(img)
        dx=cv2.Sobel(innp,bla,1,0)
        dy=cv2.Sobel(innp,bla,0,1)
        mag=np.sqrt(dx**2+dy**2)
        mag=torch.Tensor(mag.swapaxes(0,-1).swapaxes(1,2))
        p=4
        image2=rearrange(mag, 'c (p1 w) (p2 h) -> (p1 p2) w h c', p1=p, p2=p)
        image2=image2.numpy()
        tempor=[]
        for kk in range(len(image2)):
            tempor.append(np.mean(image2[kk]))
        tempor=np.asarray(tempor)
        tempor=tempor/sum(tempor)
        return tempor
    def __getitem__(self, index):
        (img, label) = self.dataset_train[index]
        if self.transformations is not None:
            return self.transformations(img), label, self.probs[index]
        return img, label
    def __len__(self):
        return len(self.dataset_train)

def balls_train(batch_size=64, resize=224):
    transform = transforms.Compose([
            transforms.Resize((resize,resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
        )
    train_data=Balls(transformations=transform, dataset_path = './data/archive(3)/train')
        #train_ds = Subset(train_data, indices[0: len(train_data)])
    print(f'Train data size {len(train_data)}')
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader
def balls_test(batch_size=64,resize=224):
    test_changes = transforms.Compose([transforms.Resize((resize,resize)),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))])
    test_data = datasets.ImageFolder('./data/archive(3)/test', transform= test_changes)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset

def build_dataset_zero_shot(is_train, args):
    transform = build_transform(False, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
