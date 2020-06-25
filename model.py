# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:21:23 2020

@author: Marco
"""

import torch.nn as nn
import torch


class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.ffn1 = nn.Linear(self.input_size, self.hidden_size)
        self.ffn2 = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        
        x = self.dropout(self.relu(self.ffn1(x)))
        x = self.ffn2(x)
        
        return x
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.n_layers)#, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, hidden):

        out, hidden = self.rnn(x, hidden)
        out = out.permute(1,0,2) #(batch_size, seq_len, hidden_size)
        out = out[:,-1,:] #Obtain only the last output 
        out = out.view(x.size()[1], -1) #(batch_size, hidden_size)
        out = self.dropout(out)
        out = self.fc(out)
        #out = self.dropout(out)
        out = out.squeeze()
        
        return out

    def initHidden(self, batch_size):
        return (torch.randn(self.n_layers, batch_size, self.hidden_size)*0.1)







