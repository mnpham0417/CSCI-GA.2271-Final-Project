import numpy as np
import torchvision.models as models
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.nn import functional as F
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp
import argparse

def build_parser():
    parser = argparse.ArgumentParser(description='Perform FID computation')
    parser.add_argument('--dataset', type=str, default='', help='Path to dataset')
    parser.add_argument('--model_path', type=str, default='', help='Path to model weights')
    parser.add_argument('--cuda', type=int, default=0, help='use gpu or not')

    return parser

# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = mean(sum_kl_d)
    # undo the logs
    is_score = exp(avg_kl_d)
    return is_score
 
if __name__ == '__main__':
    parser = build_parser()
    cuda = parser.cuda
    train_root = parser.dataset
    model_path = parser.model_path

    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((299,299)),
                                                torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(root=train_root, transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10, drop_last=True)

    model = models.inception_v3(pretrained=False, aux_logits=False, transform_input=False, num_classes=2)

    model.load_state_dict(torch.load(model_path))

    if cuda:
        model.cuda()

    model.eval()

    prob_array = []
    with torch.no_grad():
        for step, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target.reshape(len(target)))
            output = model(data)

            output = F.softmax(output, dim=1).cpu().data.numpy()
            prob_array.extend(list(output))
            if(step==200):
                break

    # conditional probabilities for low quality images
    p_yx = asarray(prob_array)
    score = calculate_inception_score(p_yx)
    print(score)