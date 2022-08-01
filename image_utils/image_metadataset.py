
import numpy as np
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets,transforms

import copy
import csv
import os

import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms as trans



##########################################################################################################################
    # data preprocessing/loaders for cifar10
##########################################################################################################################

def cifar10_metadataset(dir_root='../data',b_size=4):
    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(dir_root, train=True, download=False,
                       transform=transforms.ToTensor()),
        batch_size=b_size, shuffle=False, **kwargs)
    eval_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(dir_root, train=False, transform=transforms.ToTensor()),
        batch_size=b_size, shuffle=False, **kwargs)
    
    return train_loader,eval_loader


def get_context_idx(N, order_pixels=False):
    if order_pixels:
        idx = range(N)
    else:
        idx = random.sample(range(0, 1024), N)
    idx = torch.tensor(idx).cuda()
    return idx


def generate_grid(h, w):
    rows = torch.linspace(0, 1, h).cuda()
    cols = torch.linspace(0, 1, w).cuda()
    grid = torch.stack([cols.repeat(h, 1).t().contiguous().view(-1), rows.repeat(w)], dim=1)
    grid = grid.unsqueeze(0)
    return grid


def idx_to_y(idx, data):
    y = torch.index_select(data, dim=1, index=idx)
    return y


def idx_to_x(idx, batch_size):
    x_grid = generate_grid(32, 32)
    x = torch.index_select(x_grid, dim=1, index=idx)
    x = x.expand(batch_size, -1, -1)
    return x
    

def ravel_index(x, y, img_size):
    x = x / img_size[1]
    y = y % img_size[1]
    return x, y


def unravel_index(x, y, img_size):
    return ((x - 1) * img_size[1] + y).long()


