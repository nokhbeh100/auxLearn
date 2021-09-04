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
#%%

def sampleToBatch(example):
    example = example.reshape((1,)+example.shape)
    return example

def batchToSample(example):
    return example.reshape(example.shape[1:])


def toCuda(example):
    if CUDA:
        return example.cuda()
    else:
        return example.cpu()

# object for spliting datasets, (instead of using sklearn split_train_test)
class segmentedDataset(data.Dataset):
    def __init__(self, motherDataset, start, end, perm):
        self.mother = motherDataset
        self.start = start
        self.end = end
        self.perm = perm
    def __len__(self):
        return self.end - self.start
    def __getitem__(self, i):
        return self.mother[self.perm[self.start+i]]
#%%
cacheDatasetCounter = 0
class cacheDataset(data.Dataset):
    def __init__(self, motherDataset):
        self.mother = motherDataset
        global cacheDatasetCounter
        self.cachefolder = 'cache_' + str(cacheDatasetCounter); cacheDatasetCounter += 1
        if not(os.path.exists(self.cachefolder)):
            os.makedirs(self.cachefolder)
        
    def __len__(self):
        return len(self.mother)
    
    def __getitem__(self, i):
        cachePath = os.path.join(self.cachefolder, f'{i}.npy')
        if os.path.exists(cachePath):
            return torch.load(cachePath)
        else:
            obj = self.mother[i]
            torch.save(obj, cachePath)
            return obj

#%%
class hardNegDataset(data.Dataset):
    '''class written for hard negative training, 
    DO NOT wrap a casher around it! since things change in here (but you can wrap a casher in it :D).'''
    def __init__(self, motherDataset, train_size):
        self.mother = motherDataset
        self.idx = self.test_idx = np.arange(0, len(motherDataset))
        self.train_idx = self.test_idx[:train_size]
        self.train_size = train_size
        
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, sampleNo):
        return self.mother[self.idx[sampleNo]]
    
    def apply_hardNeg(self, errVal):
        self.train_idx = self.test_idx[ np.argsort(errVal)[-self.train_size:] ]
        
    def eval(self):
        self.idx = self.test_idx
        
    def train(self):
        self.idx = self.train_idx

#%%
def trainTestValid(mother, p1, p2, p3, seed=None):
    perm = list(range(len(mother)))
    if type(seed) is int:
        random.seed(seed)    
    random.shuffle(perm)
    s1 = segmentedDataset(mother, 0, round(p1*len(mother)), perm)
    s2 = segmentedDataset(mother, round(p1*len(mother)), round((p1+p2)*len(mother)), perm)
    s3 = segmentedDataset(mother, round((p1+p2)*len(mother)), len(mother), perm)
    return s1,s2,s3

#%% epoch timing function
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


#%% accuracy calculations for a single epoch

def calc_acc(criterion, outputs, labels):
    if type(criterion) is torch.nn.modules.loss.CrossEntropyLoss:            
        _, predicted = torch.max(outputs.data, 1)
        if CUDA:
            return (predicted.cpu()==labels.cpu()).sum().item()
        else:
            return (predicted==labels).sum().item()
            
    if (type(criterion) is torch.nn.modules.loss.BCELoss) or (type(criterion) is torch.nn.modules.loss.MSELoss):
        predicted = torch.round(outputs.data)
        if CUDA:
            return (predicted.cpu()==labels.cpu()).sum().item()
        else:
            return (predicted==labels).sum().item()
    
    return 0


#%% automatic training a single epoch
def train(model, iterator, optimizer, criterion, hardNegCriterion = None):
    epoch_loss = 0
    epoch_acc = 0
    total = 0
    model.train()
    if hardNegCriterion:
        iterator.dataset.train()
        
    bar = progressbar.ProgressBar(max_value=len(iterator))
    for i, (inputs,labels) in enumerate(iterator, 0):
        # get the inputs; data is a list of [inputs, labels]
        bar.update(i)
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()
        else:
            inputs = inputs.cpu()
            labels = labels.cpu()        
          
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        if (type(criterion) is torch.nn.modules.loss.MSELoss):
            labels = labels.to(outputs.dtype)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # do necessary accuracy calculation based on output mode
        epoch_acc += calc_acc(criterion, outputs, labels)
                    
        total += labels.size(0)

        

        # print statistics
        epoch_loss += loss.item()
    bar.finish()
    
    # if we have hard negative mining then the dataset should be fixed accordingly
    if hardNegCriterion:
        n1 = len(iterator.dataset)
        iterator.dataset.eval()
        n2 = len(iterator.dataset)
        # comparing the training and evaluation sizes
        if n1 != n2:            
            model.eval()
            print('hard Negative adaptation:')
            with torch.no_grad():
                bar = progressbar.ProgressBar(max_value=len(iterator))
                errVal = []
                for i, (images,labels) in enumerate(iterator, 0):
                    bar.update(i)
        
                    if CUDA:
                        images = images.cuda()
                        labels = labels.cuda()
                    else:
                        images = images.cpu()
                        labels = labels.cpu()        
        
                    outputs = model(images)
                    if (type(hardNegCriterion) is torch.nn.modules.loss.MSELoss):
                        labels = labels.to(outputs.dtype)                
                    
                    loss = hardNegCriterion(outputs, labels)
                    epoch_loss += torch.mean(loss)
                    errVal.append(loss.cpu().numpy())
                    total += labels.size(0)      
                # applying the error function on hard negative enabled dataset
                iterator.dataset.apply_hardNeg(np.concatenate(errVal))
                bar.finish()
        
        iterator.dataset.train()
    
    return epoch_loss / total, epoch_acc / total

#%% evalute the samples
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    total = 0
    
    if CUDA:
        model = model.cuda()
    model.eval()
    with torch.no_grad():
    
        bar = progressbar.ProgressBar(max_value=len(iterator))
        for i, (images,labels) in enumerate(iterator, 0):
            bar.update(i)

            if CUDA:
                images = images.cuda()
                labels = labels.cuda()
            else:
                images = images.cpu()
                labels = labels.cpu()        

            outputs = model(images)
            if (type(criterion) is torch.nn.modules.loss.MSELoss):
                labels = labels.to(outputs.dtype)
              
            epoch_acc += calc_acc(criterion, outputs, labels)

            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            total += labels.size(0)

        bar.finish()    
    return epoch_loss / total, epoch_acc / total

        

#%% train for a fixed number of epochs with stopping if not improving in a sequence of smart_stop epochs
def trainForEpoches(model, trainSet, testSet, optimizer, criterion, N_EPOCHS = 5, smart_stop = 100, resultFile = None, hardNegCriterion = None, bestPlace='.'):

    overfit = 0
    best_valid_loss = float('inf')

    print('CUDA=', CUDA)

    if CUDA:
        model = model.cuda()
    else:        
        model = model.cpu()


    for epoch in range(N_EPOCHS):
        
        
        start_time = time.time()
        
        print('\ntraining:')
        train_loss, train_acc = train(model, trainSet, optimizer, criterion, hardNegCriterion = hardNegCriterion)
        print('\nevaluating:')
        valid_loss, valid_acc = evaluate(model, testSet, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(bestPlace,'temp-best.pt') )
            overfit = 0
        else:
            overfit += 1
        print(f'\nEpoch: {epoch+1:02}/{N_EPOCHS} | Epoch Time: {epoch_mins}m {epoch_secs}s | overfit counter:{overfit}/{smart_stop}')
        print(f'\tTrain Acc: {train_acc*100:.2f}% | Train Loss: {train_loss}')
        print(f'\tVal. Acc: {valid_acc*100:.2f}% |  Val. Loss: {valid_loss}')
        if resultFile:
            resultFile.write('x')
            resultFile.flush()
        if (overfit > smart_stop):
            break
    if resultFile:
        resultFile.write('\n')
        resultFile.flush()
    model.load_state_dict(torch.load( os.path.join(bestPlace,'temp-best.pt') ))

