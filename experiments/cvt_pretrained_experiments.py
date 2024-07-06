import copy
import torchvision
import torch
import tensorflow as tf
#from tensorflow.keras.optimizers import Adamax
from torch import optim
import yaml
import arguments
import math
from data_handlers import get_train_val_loaders_cifar, get_test_loader_cifar, get_train_loader_tiny_imagenet, \
    get_valid_loader_tiny_imagenet, get_train_val_loaders_food101, get_test_loader_food101
from models.cvt import ConvolutionalVisionTransformer
from models.wide_resnet import Wide_ResNet
from fibonacci import fibonacci
from datasets import balls_train, balls_test, seaanimals_train_test, ahe_train, ahe_test
from cvt_train import TrainerCvt
#import torch_optimizer as optim
import numpy
def add_optimizer_params_lr(trainer, initial_learning_rate):
    #return None
    for name, param in trainer.model.named_parameters():
        if 'head' not in name and 'norm_final' not in name:
            trainer.optimizer.add_param_group({'params': param, 'lr': initial_learning_rate})

def build_cvt(args):
    with open("configs/cvt_configs.yaml", 'r') as stream:
        try:
            data = (yaml.safe_load(stream))
        except Exception as exc:
            print(exc)
    cvt = ConvolutionalVisionTransformer(spec=data["MODEL"]['SPEC'], num_classes=args.num_classes)
    return cvt,data

def build_optimizer_cvt(model, args):
    list_params = []
    for name, param in model.named_parameters():
        if 'head' in name or 'norm_final' in name:
            list_params.append(param)
    return optim.Adamax(list_params, lr=args.lr)

def baseline(num_epochs):
    return [0] * num_epochs


def lin_repeat_40():
    v = fibonacci(length=7)
    for i in range(1,len(v)):
        v[i]=math.log(v[i])/(math.log(v[6])/0.4036067977500615)
    v=v[2:]
    v[0] = 0.07
    return v

def lin_repeat_60():
    v = fibonacci(length=7) #7
    for i in range(1,len(v)):
        v[i]=math.log(v[i])/(math.log(v[6])/0.6036067977500615) #0.6 6
    v=v[2:]
    v[0] = 0.1
    return v
def train_cifar10(args):
    args.num_classes = 10
    args.num_epochs = 40
    args.model_name = 'cvt_c10'
    cvt,configs = build_cvt(args)
    state_dict = torch.load(f"./pretrained_models/CvT-13-224x224-IN-1k.pth")
    state_dict['norm_final.weight'] = state_dict['norm.weight']
    state_dict['norm_final.bias'] = state_dict['norm.bias']
    del state_dict['norm.weight']
    del state_dict['norm.bias']
    state_dict['head.weight'] = cvt.state_dict()['head.weight']
    state_dict['head.bias'] = cvt.state_dict()['head.bias']
    cvt.load_state_dict(state_dict)

    curriculum = lin_repeat_40()
    args.percent = curriculum * 8

    print(args.percent)
    print(len(args.percent))
    train_loader, val_loader = get_train_val_loaders_cifar(val_size=64, dataset=torchvision.datasets.CIFAR10, resize=64)
    test_loader = get_test_loader_cifar(dataset=torchvision.datasets.CIFAR10, resize=64)
    args.testlo=test_loader
    trainer = TrainerCvt(cvt, train_loader, test_loader, add_optimizer_params_lr,  args,build_optimizer_cvt,configs)
    trainer.train()

def train_food101(args):
    args.num_classes = 101
    args.num_epochs = 40
    args.model_name = 'cvt_food101b'
    cvt,configs = build_cvt(args)
    state_dict = torch.load(f"./pretrained_models/CvT-13-224x224-IN-1k.pth")
    state_dict['norm_final.weight'] = state_dict['norm.weight']
    state_dict['norm_final.bias'] = state_dict['norm.bias']
    del state_dict['norm.weight']
    del state_dict['norm.bias']
    state_dict['head.weight'] = cvt.state_dict()['head.weight']
    state_dict['head.bias'] = cvt.state_dict()['head.bias']
    cvt.load_state_dict(state_dict)

    curriculum = lin_repeat_60() #era 60
    args.percent = curriculum * 8 #8
    #args.percent = baseline(40)
    print(args.percent)
    print(len(args.percent))
    train_loader = get_train_val_loaders_food101(dataset=torchvision.datasets.Food101, resize=224)
    test_loader = get_test_loader_food101(dataset=torchvision.datasets.Food101, resize=224)
    args.testlo=test_loader
    trainer = TrainerCvt(cvt, train_loader, test_loader, add_optimizer_params_lr,  args,build_optimizer_cvt,configs)
    trainer.train()
def train_seaanimals(args):
    args.num_classes = 23
    args.num_epochs = 40
    args.model_name = 'cvt_sea'
    cvt,configs = build_cvt(args)
    state_dict = torch.load(f"./pretrained_models/CvT-13-224x224-IN-1k.pth")
    state_dict['norm_final.weight'] = state_dict['norm.weight']
    state_dict['norm_final.bias'] = state_dict['norm.bias']
    del state_dict['norm.weight']
    del state_dict['norm.bias']
    state_dict['head.weight'] = cvt.state_dict()['head.weight']
    state_dict['head.bias'] = cvt.state_dict()['head.bias']
    cvt.load_state_dict(state_dict)

    curriculum = lin_repeat_60() #era 60
    args.percent = curriculum * 8 #8
    #args.percent = baseline(40)
    print(args.percent)
    print(len(args.percent))
    train_loader ,test_loader = seaanimals_train_test(resize=224)

    args.testlo=test_loader
    trainer = TrainerCvt(cvt, train_loader, test_loader, add_optimizer_params_lr,  args,build_optimizer_cvt,configs)
    trainer.train()
def ahe_train_1(args):
    args.num_classes = 10
    args.num_epochs = 40
    args.model_name = 'cvt_ahe'
    cvt,configs = build_cvt(args)
    state_dict = torch.load(f"./pretrained_models/CvT-13-224x224-IN-1k.pth")
    state_dict['norm_final.weight'] = state_dict['norm.weight']
    state_dict['norm_final.bias'] = state_dict['norm.bias']
    del state_dict['norm.weight']
    del state_dict['norm.bias']
    state_dict['head.weight'] = cvt.state_dict()['head.weight']
    state_dict['head.bias'] = cvt.state_dict()['head.bias']
    cvt.load_state_dict(state_dict)

    curriculum = lin_repeat_60() #era 60
    args.percent = curriculum * 8 #8
    #args.percent = baseline(40)
    print(args.percent)
    print(len(args.percent))
    train_loader = ahe_train(resize=128)
    test_loader = ahe_test(resize=128)
    args.testlo=test_loader
    trainer = TrainerCvt(cvt, train_loader, test_loader, add_optimizer_params_lr,  args,build_optimizer_cvt,configs)
    trainer.train()

def train_balls(args):
    args.num_classes = 15
    args.num_epochs = 40
    args.model_name = 'cvt_food101b'
    cvt,configs = build_cvt(args)
    state_dict = torch.load(f"./pretrained_models/CvT-13-224x224-IN-1k.pth")
    state_dict['norm_final.weight'] = state_dict['norm.weight']
    state_dict['norm_final.bias'] = state_dict['norm.bias']
    del state_dict['norm.weight']
    del state_dict['norm.bias']
    state_dict['head.weight'] = cvt.state_dict()['head.weight']
    state_dict['head.bias'] = cvt.state_dict()['head.bias']
    cvt.load_state_dict(state_dict)

    curriculum = lin_repeat_60() #era 60
    args.percent = curriculum * 8 #8
    #args.percent = baseline(40)
    print(args.percent)
    print(len(args.percent))
    train_loader = balls_train(resize=224)
    test_loader = balls_test(resize=224)
    args.testlo=test_loader
    trainer = TrainerCvt(cvt, train_loader, test_loader, add_optimizer_params_lr,  args,build_optimizer_cvt,configs)
    trainer.train()
def train_cifar100(args):
    args.num_classes = 100
    args.num_epochs = 40
    args.model_name = 'cvt_c100'
    cvt,configs = build_cvt(args)
    state_dict = torch.load(f"./pretrained_models/CvT-13-224x224-IN-1k.pth")
    state_dict['norm_final.weight'] = state_dict['norm.weight']
    state_dict['norm_final.bias'] = state_dict['norm.bias']
    del state_dict['norm.weight']
    del state_dict['norm.bias']
    state_dict['head.weight'] = cvt.state_dict()['head.weight']
    state_dict['head.bias'] = cvt.state_dict()['head.bias']
    cvt.load_state_dict(state_dict)

    curriculum = lin_repeat_60()
    args.percent = curriculum * 8 #8
    print(args.percent)
    print(len(args.percent))
    train_loader, val_loader = get_train_val_loaders_cifar(val_size=64, dataset=torchvision.datasets.CIFAR100, resize=64)
    test_loader = get_test_loader_cifar(dataset=torchvision.datasets.CIFAR100, resize=64)
    args.testlo=test_loader
    trainer = TrainerCvt(cvt, train_loader, test_loader, add_optimizer_params_lr,  args,build_optimizer_cvt,configs)
    trainer.train()


def train_tinyimagenet(args):
    args.num_classes = 200
    args.num_epochs = 40
    args.model_name = 'cvt_tin'
    cvt,configs = build_cvt(args)
    state_dict = torch.load(f"./pretrained_models/CvT-13-224x224-IN-1k.pth")
    state_dict['norm_final.weight'] = state_dict['norm.weight']
    state_dict['norm_final.bias'] = state_dict['norm.bias']
    del state_dict['norm.weight']
    del state_dict['norm.bias']
    state_dict['head.weight'] = cvt.state_dict()['head.weight']
    state_dict['head.bias'] = cvt.state_dict()['head.bias']
    cvt.load_state_dict(state_dict)

    curriculum = lin_repeat_60()
    args.percent = curriculum * 8
    print(args.percent)
    print(len(args.percent))
    train_loader, class_to_idx = get_train_loader_tiny_imagenet(path="./data")
    val_loader = get_valid_loader_tiny_imagenet(path="./data", class_to_idx=class_to_idx)
    args.testlo=val_loader
    trainer = TrainerCvt(cvt, train_loader, val_loader, add_optimizer_params_lr,  args,build_optimizer_cvt,configs)
    trainer.train()