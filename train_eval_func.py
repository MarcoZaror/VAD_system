# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:36:01 2020

@author: Marco
"""
import numpy as np
import torch
from sklearn.metrics import roc_curve, accuracy_score
import torch.nn as nn

def pred(value):
  if value < 0.5:
    return 0
  else:
    return 1

def compute_eer(fpr,tpr,thresholds):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, thresholds[min_index]
   
def train_ffnn(loader, model, device, criterion, optimizer):
    model.train()
    loss = 0
    batches = 0
    for input, labels in loader:
        input = input.to(device)
        labels = labels.to(device)
        labels = labels.long()
        optimizer.zero_grad()
        preds = model(input).squeeze()
        loss_batch = criterion(preds,labels)
        loss_batch.backward()
        optimizer.step()
        loss += loss_batch
        batches += 1
    return loss/batches 

def val_ffnn(loader, model, device, criterion):
    model.eval()
    loss_t = 0
    batches_t = 0
    with torch.no_grad():
        for input, labels in loader:
            input = input.to(device)
            labels = labels.long()
            labels = labels.to(device)
            preds = model(input).squeeze()
            loss_batch_t = criterion(preds,labels)
            loss_t += loss_batch_t
            batches_t += 1
    return loss_t/batches_t

def eer_acc_ffnn(loader, model, device):
    preds = []
    labels_list = []
    with torch.no_grad():
        for input, labels in loader:
            input = input.to(device)
            p = model(input).squeeze()
            preds.append(p[:,1].flatten())
            labels_list.append(labels.long())
        preds = [item.cpu().numpy() for sublist in preds for item in sublist]
        l_test = [int(item.cpu()) for sublist in labels_list for item in sublist]
        fpr,tpr,thresholds = roc_curve(l_test,preds)   #true, score #labels_test.detach().numpy()
        eer,_ = compute_eer(fpr,tpr,thresholds)
        preds = [pred(item) for item in preds]
        acc = accuracy_score(l_test, preds)
        return eer, acc

def det_ffnn(loader, model, device):
    preds = []
    labels_list = []
    with torch.no_grad():
        for input, labels in loader:
            input = input.to(device)
            p = model(input).squeeze()
            preds.append(p[:,1].flatten())
            labels_list.append(labels.long())
        preds = [item.cpu().numpy() for sublist in preds for item in sublist]
        l_test = [int(item.cpu()) for sublist in labels_list for item in sublist]
        fpr,tpr,thresholds = roc_curve(l_test,preds)   #true, score #labels_test.detach().numpy()
        fnr = 1-tpr
        eer,_ = compute_eer(fpr,tpr,thresholds)
        preds = [pred(item) for item in preds]
        acc = accuracy_score(l_test, preds)
        return fpr, fnr, eer, acc
    
def train_rnn(loader, model, device, criterion, optimizer):
    model.train()
    loss = 0
    batches = 0
    for input, labels in loader:
        input = input.to(device)
        labels = labels.to(device)
        input = input.permute(1, 0, 2)
        batch_size = input.size()[1]
        optimizer.zero_grad()
        hidden = model.initHidden(batch_size)
        hidden = hidden.to(device)
        preds = model(input, hidden)
        loss_batch = criterion(preds,labels.long())
        loss_batch.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        loss += loss_batch
        batches += 1
    return loss/batches 

def val_rnn(loader, model, device, criterion):
    model.eval()
    loss = 0
    batches = 0
    with torch.no_grad():
        for input, labels in loader:
            input = input.to(device)
            labels = labels.to(device)
            input = input.permute(1, 0, 2)
            batch_size = input.size()[1]
            hidden = model.initHidden(batch_size)
            hidden = hidden.to(device)
            preds = model(input, hidden)
            loss_batch = criterion(preds,labels.long())
            loss += loss_batch
            batches += 1
    return loss/batches 

def eer_acc_rnn(loader, model, device):
    preds = []
    labels_list = []
    with torch.no_grad():
        for input, labels in loader:
            input = input.to(device)
            input = input.permute(1, 0, 2)
            batch_size = input.size()[1]
            hidden = model.initHidden(batch_size)
            hidden = hidden.to(device)
            p = model(input, hidden)
            preds.append(p[:,1].flatten())
            labels_list.append(labels.long())
        
        preds = [item.cpu().numpy() for sublist in preds for item in sublist]
        l_test = [int(item.cpu()) for sublist in labels_list for item in sublist]
        
        fpr,tpr,thresholds = roc_curve(l_test,preds)   #true, score #labels_test.detach().numpy()
        eer,_ = compute_eer(fpr,tpr,thresholds)
        preds = [pred(item) for item in preds]
        acc = accuracy_score(l_test, preds)
        return eer, acc

def det_rnn(loader, model, device):
    preds = []
    labels_list = []
    with torch.no_grad():
        for input, labels in loader:
            input = input.to(device)
            input = input.permute(1, 0, 2)
            batch_size = input.size()[1]
            hidden = model.initHidden(batch_size)
            hidden = hidden.to(device)
            p = model(input, hidden)
            preds.append(p[:,1].flatten())
            labels_list.append(labels.long())
            
        preds = [item.cpu().numpy() for sublist in preds for item in sublist]
        l_test = [int(item.cpu()) for sublist in labels_list for item in sublist]
        fpr,tpr,thresholds = roc_curve(l_test,preds)   #true, score #labels_test.detach().numpy()
        fnr = 1-tpr
        eer,_ = compute_eer(fpr,tpr,thresholds)
        preds = [pred(item) for item in preds]
        acc = accuracy_score(l_test, preds)
        return fpr, fnr, eer, acc