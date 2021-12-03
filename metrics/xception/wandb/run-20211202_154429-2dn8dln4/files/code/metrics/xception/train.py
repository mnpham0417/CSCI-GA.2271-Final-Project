'''
Training script for xception
'''

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
from torch.autograd import Variable
import argparse
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import torchvision.models as models
import random
import copy
import torch.nn.functional as F
matplotlib.use('Agg') # For headless servers
import wandb
from xception import *

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def build_parser():
    parser = argparse.ArgumentParser(description='Training script for xception')
    parser.add_argument("--batch_size", type=int, default=64, help="training and testing batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate for models")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--num_epoch", type=int, default=100, help="number of epochs")
    parser.add_argument("--cuda", type=int, default=1, help="use gpu or not")
    parser.add_argument("--wandb_name", type=str, default="test", help="name for experiment on wandb")
    parser.add_argument("--data_dir", type=str, default="/media/mnpham/HARD_DISK_2/dataset/self-distillation", help="folder of data")
    parser.add_argument("--image_size", type=int, default=64, help="size of training/testing images")
    parser.add_argument("--separate_validation", type=int, default=0, help="if False then use training as testing data")
    return parser

def build_model(pretrained=False):
    return xception(pretrained)

def train(model, optimizer, criterion, train_loader, test_loader):
    wandb.init(name=args.wandb_name,project='graduate_school', entity='mnphamx1', reinit=True)
    for epoch in range(args.num_epoch):
        dict_wandb = {}
        model.train()
        training_loss = 0
        correct = 0
        for idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            print(x.shape, y.shape)
            assert 1 == 0
            if(args.cuda):
                x = x.cuda()
                y = y.cuda()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            pred = y_pred.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            print(idx, len(train_loader))
     

if __name__ == '__main__':
    args = build_parser().parse_args()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    # training_root = os.path.join(args.data_dir, "training")
    # if(args.separate_validation):
    #     validation_root = os.path.join(args.data_dir, "validation")
    # else:
    #     validation_root = os.path.join(args.data_dir, "training")

    # training_loader = torchvision.datasets.ImageFolder(root=training_root, transform=transform)
    # validation_loader = torchvision.datasets.ImageFolder(root=validation_root, transform=transform)

    #model
    model = build_model()

    #loss
    criterion = nn.CrossEntropyLoss()

    #optimizer
    optimizer = optim.SGD(model.parameters() , lr=args.learning_rate, momentum=0.9)

    imagenet_train_root = os.path.join(args.data_dir, "imagenet/train")
    imagenet_val_root = os.path.join(args.data_dir, "imagenet/val")
    data_train = torchvision.datasets.ImageFolder(root=imagenet_train_root, transform=transform)
    data_val = torchvision.datasets.ImageFolder(root=imagenet_val_root, transform=transform)
    data_train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    data_val_loader = DataLoader(dataset=data_val, batch_size=args.batch_size, shuffle=False)

    train(model, optimizer, criterion, data_train_loader, data_val_loader)

