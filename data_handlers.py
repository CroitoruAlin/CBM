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
import random
from einops.layers.torch import Rearrange
from einops import rearrange
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#from models.resnet import MaskIn

class Food101(torch.utils.data.Dataset):
    def __init__(self, dataset_path='./data', transformations=None, should_download=True,data_set=torchvision.datasets.Food101):
        self.dataset_train = data_set(dataset_path, download=should_download,split = "train")
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

def get_train_val_loaders_food101(batch_size=64, dataset=torchvision.datasets.Food101,resize=32):
    transform = transforms.Compose([
        transforms.Resize((resize,resize)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    )
    train_data=Food101(transformations=transform, data_set=dataset)
    #train_ds = Subset(train_data, indices[0: len(train_data)])
    print(f'Train data size {len(train_data)}')
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader

def get_test_loader_food101(batch_size=64, dataset=torchvision.datasets.CIFAR10,resize=32):
    test_changes = transforms.Compose([transforms.Resize((resize,resize)),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))])
    test_data = dataset(root='./data', split='test', download=True, transform=test_changes)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader

class CIFAR(torch.utils.data.Dataset):
    def __init__(self, dataset_path='./data', transformations=None, should_download=True,data_set=torchvision.datasets.CIFAR10):
        self.dataset_train = data_set(dataset_path, download=should_download)
        self.transformations = transformations
        self.probs=[]
        for i in range(len(self.dataset_train)):
            (imeg, laebl) = self.dataset_train[i]
            self.probs.append(self.get_probability(imeg))
    def get_probability(self, img):
        bla=cv2.CV_64F
        
        transformz = transforms.Compose([
        transforms.Resize(64)] # adjust to 32 for non cvt
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
def get_train_val_loaders_cifar(val_size=2500, batch_size=64, dataset=torchvision.datasets.CIFAR10,resize=32):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    )
    val_transforms = [transforms.ToTensor(),        transforms.Resize(resize),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    val_changes = transforms.Compose(val_transforms)
    train_data=CIFAR(transformations=transform, data_set=dataset)
    val_data = dataset(root='./data', train=True, download=True, transform=val_changes)
    torch.manual_seed(33)
    train_size = len(train_data) - val_size
    indices = randperm(sum([train_size, val_size]), generator=default_generator).tolist()
    train_ds = Subset(train_data, indices[0: train_size])
    val_ds = Subset(val_data, indices[train_size: train_size + val_size])
    print(f'Train data size {len(train_ds)}, Validation data size {len(val_ds)}')
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def get_test_loader_cifar(batch_size=64, dataset=torchvision.datasets.CIFAR10,resize=32):
    test_changes = transforms.Compose([transforms.Resize(resize),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))])
    test_data = dataset(root='./data', train=False, download=True, transform=test_changes)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader


def get_train_loader_tiny_imagenet(path, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    )
    train_data = TinyImagenetTrain(
        os.path.join(path, 'tiny-imagenet-200', 'train'),
        transform=transform
    )
    return DataLoader(
        train_data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    ), train_data.class_to_idx



def get_valid_loader_tiny_imagenet(path, class_to_idx, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
    )
    train_data = TinyImagenetDatasetValidation(path, class_to_idx, transform=transform)
    
    return DataLoader(
        train_data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )
    



class TinyImagenetDatasetValidation(Dataset):
    def __init__(self, path, class_to_idx, transform=None):
        annotations = os.path.join(path, 'tiny-imagenet-200', 'val', "val_annotations.txt")
        val_annotations = pd.read_csv(annotations, sep='\t', lineterminator='\n', header=None,
                                      names=['file_name', 'id', 'ignore1', 'ignore2', 'ignore3', 'ignore4'],
                                      encoding='utf-8', quoting=csv.QUOTE_NONE)

        file_to_class = {}
        self.list_images = []
        for _, elem in val_annotations.iterrows():
            file_to_class[elem['file_name']] = class_to_idx[elem['id']]
            self.list_images.append(elem['file_name'])
        self.file_to_class = file_to_class
        self.path = os.path.join(path, 'tiny-imagenet-200', 'val', 'images')
        self.transform = transform

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, item):
        file_name = os.path.join(self.path, self.list_images[item])
        image = default_loader(file_name)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.file_to_class[self.list_images[item]]



class TinyImagenetTrain(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        # self.result={}
        self.images=[]
        self.probs=[]

        for path,target in self.samples:
            sample = self.loader(path)
            self.images.append((sample,target))
            self.probs.append(self.get_probability(sample))

    def get_probability(self, img):
        bla=cv2.CV_64F
        #innp=torch.Tensor.numpy(img.swapaxes(0,-1).swapaxes(0,1))
        innp=np.asarray(img)
        dx=cv2.Sobel(innp,bla,1,0)
        dy=cv2.Sobel(innp,bla,0,1)
        mag=np.sqrt(dx**2+dy**2)
        mag=torch.Tensor(mag.swapaxes(0,-1).swapaxes(1,2))
        image2=rearrange(mag, 'c (p1 w) (p2 h) -> (p1 p2) w h c', p1=4, p2=4)
        image2=image2.numpy()
        tempor=[]
        for kk in range(len(image2)):
            tempor.append(np.mean(image2[kk]))
        tempor=np.asarray(tempor)
        tempor=tempor/sum(tempor)
        return tempor

    def __getitem__(self, item):
        image,target = self.images[item]
        if self.transform is not None:
            #prob = self.get_probability(image)
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target, self.probs[item]
