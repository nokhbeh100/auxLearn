#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 19:27:24 2019

@author: mnz
"""

import torch
import time
import random
import numpy as np
import torchvision
from matplotlib import pyplot as plt
import os

# forcing to not use cuda or use cuda
CUDA = torch.cuda.is_available();False;
device = ('cuda:0' if CUDA else 'cpu')
import torch.utils.data as data
import progressbar
from PIL import Image

from auxLearn import *
#%%

# loading dataset based on a folder and spliting at the same time, mainly used for image datasets
def loadDataset(folder, transform, p1=.6, p2=.2, p3=.2, loader=plt.imread, shuffle=False, num_workers=4, batch_size=4):
    
    dataset = torchvision.datasets.ImageFolder(folder, loader=plt.imread, transform=transform )
    
    trainSet, testSet, validSet = trainTestValid( dataset, p1, p2, p3)
    
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)
    
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)
    
    validLoader = torch.utils.data.DataLoader(validSet, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)
    return trainLoader, testLoader, validLoader



def resize(img, size=(227,227)):
    return np.array(Image.fromarray(img.astype(np.double)).resize(size, resample=Image.BOX), dtype='double')

def keepAspectResize(img, size=(227,227)):
    ratio = min(size[0]/img.shape[0], size[1]/img.shape[1])
    return np.array(Image.fromarray(img.astype(np.double)).resize((int(ratio*img.shape[0]), int(ratio*img.shape[1])), resample=Image.BOX), dtype='double')

def gray2rgb(x):
    if len(x.shape) == 3:
        if x.shape[-1] == 4: #RGBA to RGB
            x = x[:,:, :3]
    if len(x.shape) == 2:
        x = np.repeat(x.reshape(x.shape+(1,)), 3, axis=2)
    return x



