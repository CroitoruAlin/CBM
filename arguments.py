import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='balls',
                        choices=['cifar10', 'cifar100', 'tinyimagenet', 'food101','balls','seaanimals','ahe'])
    parser.add_argument('--model_name', type=str, default='cvt_pretrained',
                        choices=['resnet18', 'wresnet', 'cvt_pretrained'])
    parser.add_argument('--lr', type=float, default=5e-4)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args
