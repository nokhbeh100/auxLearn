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
def trainTestValid(mother, p1, p2, p3):
    perm = list(range(len(mother)))
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



#%% automatic training a single epoch
def train(model, iterator, optimizer, criterion, hardNeg = False):
    epoch_loss = 0
    epoch_acc = 0
    total = 0
    model.train()
    if hardNeg:
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
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # do necessary accuracy calculation based on output mode
        if type(criterion) is torch.nn.modules.loss.CrossEntropyLoss:            
            _, predicted = torch.max(outputs.data, 1)
            if CUDA:
                epoch_acc += (predicted.cpu()==labels.cpu()).sum().item()
            else:
                epoch_acc += (predicted==labels).sum().item()
                
        if type(criterion) is torch.nn.modules.loss.BCELoss:
            predicted = torch.round(outputs.data)
            if CUDA:
                epoch_acc += (predicted.cpu()==labels.cpu()).sum().item()
            else:
                epoch_acc += (predicted==labels).sum().item()
            
        total += labels.size(0)

        

        # print statistics
        epoch_loss += loss.item()
    bar.finish()
    
    # if we have hard negative mining then the dataset should be fixed accordingly
    if hardNeg:
        n1 = len(iterator.dataset)
        iterator.dataset.eval()
        n2 = len(iterator.dataset)
        # comparing the training and evaluation sizes
        if n1 != n2:            
            model.eval()
            print('hard Negative adaptation:')
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
                  
                if type(criterion) is torch.nn.modules.loss.CrossEntropyLoss:            
                    _, predicted = torch.max(outputs.data, 1)
                    if CUDA:
                        epoch_acc += (predicted.cpu()==labels.cpu()).sum().item()
                    else:
                        epoch_acc += (predicted==labels).sum().item()
                        
                if type(criterion) is torch.nn.modules.loss.BCELoss:
                    predicted = torch.round(outputs.data)
                    if CUDA:
                        epoch_acc += (predicted.cpu()==labels.cpu()).sum().item()
                    else:
                        epoch_acc += (predicted==labels).sum().item()
    
                loss = criterion(outputs, labels)
                epoch_loss += loss.item()
                errVal.append(loss.item())
                total += labels.size(0)      
            # applying the error function on hard negative enabled dataset
            iterator.dataset.apply_hardNeg(errVal)
        
        iterator.dataset.train()
    
    return epoch_loss / total, epoch_acc / total

#%% evalute the samples
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    total = 0
    
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
              
            if type(criterion) is torch.nn.modules.loss.CrossEntropyLoss:            
                _, predicted = torch.max(outputs.data, 1)
                if CUDA:
                    epoch_acc += (predicted.cpu()==labels.cpu()).sum().item()
                else:
                    epoch_acc += (predicted==labels).sum().item()
                    
            if type(criterion) is torch.nn.modules.loss.BCELoss:
                predicted = torch.round(outputs.data)
                if CUDA:
                    epoch_acc += (predicted.cpu()==labels.cpu()).sum().item()
                else:
                    epoch_acc += (predicted==labels).sum().item()

            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            total += labels.size(0)

    bar.finish()    
    return epoch_loss / total, epoch_acc / total

        

#%% train for a fixed number of epochs with stopping if not improving in a sequence of smart_stop epochs
def trainForEpoches(model, trainSet, testSet, optimizer, criterion, N_EPOCHS = 5, smart_stop = 100, resultFile = None, hardNeg = False, bestPlace='.'):

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
        train_loss, train_acc = train(model, trainSet, optimizer, criterion, hardNeg = hardNeg)
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

