# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:04:48 2020

@author: Marco
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os

class Data2:
    def __init__(self):
        self.data = np.array([])
        self.labels = np.array([])
    
    def loadData(self, path, codes):
        files = []
        for file in os.listdir(path):
            for code in codes:
                if file[:3] == code:
                    with open (path/file, 'rb' ) as f :
                        files.append(np.load(f))
        self.data = np.concatenate((files[0], files[1]))
        for i in range(2,len(files)):
            self.data = np.concatenate((self.data,files[i]))
        return self.data
    
    def load_labels(self, path, codes):
        files = []
        for file in os.listdir(path):
            for code in codes:
                if file[:3] == code:
                    with open (path/file, 'rb' ) as f :
                        files.append(np.load(f))
        self.labels = np.concatenate((files[0], files[1]))
        for i in range(2,len(files)):
            self.labels = np.concatenate((self.labels,files[i]))
        return self.labels
    
    def addContext(self, context = 2):
        '''
        Add c previous and following examples to an example and 
        modify the labels accordingly
        '''
        f = np.array([])
        f2 = []
        for i in range(context, len(self.data)- context):
            for c in range(-context,context+1):
                f = np.concatenate((f,self.data[i+c]),axis=0)
            f2.append(f)
            f = np.array([]) # reset value
        self.data = np.array(f2)
        self.labels = self.labels[context:-context]
        return self.data, self.labels
        
    def preprocessData(self, subsample, bs):
        #Take the data and returns an iterator
        if subsample > 0:
            idx = np.random.permutation(len(self.data))
            subsample = int(subsample*len(self.data)) 
            self.data = self.data[idx]
            self.labels = self.labels[idx]
            self.data = self.data[:subsample]
            self.labels = self.labels[:subsample]
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels).squeeze().float()
        dataset = TensorDataset(self.data,self.labels)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        return loader, self.data.shape[1]

    def preprocessSeqData(self, subsample, seq_len, bs):
        #Take the data and returns an iterator
        if subsample > 0:
            self.data = self.data[:int(subsample*len(self.data))]
            self.labels = self.labels[:int(subsample*len(self.labels))]
        f = np.zeros((self.data.shape[0] - (seq_len-1), seq_len, self.data.shape[1]))
        for i in range(self.data.shape[0] - (seq_len-1)):
            f[i] = self.data[i:i+seq_len]
        l = np.zeros((self.data.shape[0] - (seq_len-1)))
        for i in range(self.labels.shape[0] - (seq_len-1)):
            l[i] = self.labels[i+(seq_len-1)] 
        f = torch.tensor(f, dtype = torch.float32)
        l = torch.tensor(l, dtype = torch.float32)
        dataset = TensorDataset(f,l)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        return loader, self.data.shape[1]
