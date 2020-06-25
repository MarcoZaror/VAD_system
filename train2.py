# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:26:10 2020

@author: Marco

File created to train a feed-forward neural network or a recurrent neural 
network. Receive as input a irectory containing the audio files, 
another containing the labels and a flag indicating to train the FFNN or RNN.

USE: python <PROGNAME>  
    - directory audio
    - directory labels
    - 1 for RNN or 0 for FFNN
    
Example for project -> python train.py audio labels 0
"""

import numpy as np
from pathlib import Path
import os
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve
from Data2 import Data2
#from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#from Data import loadData, addContext, preprocessData, preprocessSeqData
from model import FFN, RNN
from train_eval_func import pred, compute_eer, train_ffnn, val_ffnn, eer_acc_ffnn, train_rnn, val_rnn, eer_acc_rnn, det_ffnn, det_rnn
import sys
import glob
from pathlib import Path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#path_audio = Path('C:/Users/Marco/Documents/MSc/Speech/Speech Technology/Tasks/IV/audio/')
#path_labels = Path('C:/Users/Marco/Documents/MSc/Speech/Speech Technology/Tasks/IV/labels/')

path_audio = Path(sys.argv[1])
path_labels = Path(sys.argv[2])
#os.listdir('audio')
#glob.glob('audio/*.npy')
#files_dev = files_in('dev/*.probs')
# Load data

data_tr = Data2()
data_val = Data2()
data_test = Data2()

data_tr.loadData(path_audio,['NIS','VIT'])
data_tr.load_labels(path_labels,['NIS','VIT']).astype(float)
data_val.loadData(path_audio,['EDI'])
data_val.load_labels(path_labels,['EDI']).astype(float)
data_test.loadData(path_audio,['CMU'])
data_test.load_labels(path_labels,['CMU']).astype(float)

#dat2,lab2 = a.addContext(1)

#c,d = a.preprocessData(0.1, 10)
#e,f = a.preprocessSeqData(0.1, 5, 10)
print('Data loaded')

if sys.argv[3] == '0':
    # Preprocess data
    data_tr.addContext(2)
    data_val.addContext(2)
    data_test.addContext(2)
    loader_tr, input_size = data_tr.preprocessData(subsample=0.5, bs=64)
    loader_test,_ = data_val.preprocessData(subsample=0, bs=64)
    loader_val,_ = data_test.preprocessData(subsample=0, bs=64)
    
    print('Data processed')
    # Define model
    n_hidden = 150
    n_output = 2  
    ffnn = FFN(input_size, n_hidden, n_output)
    ffnn.to(device)
    
    optimizer = torch.optim.SGD(ffnn.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    
    epoch = 1
    losses = []
    losses_val = []
    it = 0
    while True:
        loss = train_ffnn(loader_tr, ffnn, device, criterion, optimizer)
        losses.append(loss)
        print('Train: Epoch {}: loss: {}'.format(epoch, loss.item()))
        loss_val = val_ffnn(loader_val, ffnn, device, criterion)
        losses_val.append(loss_val)
        print('Val: Epoch {}: loss: {}'.format(epoch, loss_val.item()))
        loss_sorted = sorted(losses_val)
        #eer_v,acc_v = eer_acc_ffnn(loader_val, ffnn, device)
        #print('Val: Epoch {}: EER: {}, ACC: {}'.format(epoch, eer_v, acc_v))
        #eer_t,acc_t = eer_acc_ffnn(loader_test, ffnn, device)
        #print('Test: Epoch {}: EER: {}, ACC: {}'.format(epoch, eer_t, acc_t))
        if (len(losses_val) >1) and (losses_val[-1] > losses_val[-2]):
            if it == 1:
                break
            else:
                it = 1
        else:
            torch.save(ffnn.state_dict(),os.path.join(os.getcwd(),'model-'+str(epoch)))
            it = 0
            epoch += 1

elif sys.argv[3] == '1': #RNN
    # Preprocess the data
    #data_tr.addContext(2)
    #data_val.addContext(2)
    #data_test.addContext(2)
    loader_tr, input_size = data_tr.preprocessSeqData(subsample=0.2,seq_len=3, bs=64)
    loader_test,_ = data_val.preprocessSeqData(subsample=0.2,seq_len=3, bs=64)
    loader_val,_ = data_test.preprocessSeqData(subsample=0.2,seq_len=3, bs=64)
    print('Data processed')
    
    # Define the model
    hidden_size = 100
    n_layers = 1
    output_size = 2
    
    rnn = RNN(input_size,hidden_size, output_size, n_layers)
    rnn = rnn.to(device)
    optimizer = torch.optim.SGD(rnn.parameters(), lr = 0.0001)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    
    epoch = 1
    losses = []
    losses_val = []
    it = 0
    while True:
        loss = train_rnn(loader_tr, rnn, device, criterion, optimizer)
        losses.append(loss)
        print('Train: Epoch {}: loss: {}'.format(epoch, loss.item()))
        loss_val = val_rnn(loader_val, rnn, device, criterion)
        losses_val.append(loss_val)
        print('Val: Epoch {}: loss: {}'.format(epoch, loss_val.item()))
        #eer_v,acc_v = eer_acc_rnn(loader_val, rnn, device)
        #print('Val: Epoch {}: EER: {}, ACC: {}'.format(epoch, eer_v, acc_v))
        #eer_t,acc_t = eer_acc_rnn(loader_test, rnn, device)
        #print('Test: Epoch {}: EER: {}, ACC: {}'.format(epoch, eer_t, acc_t))
        loss_sorted = sorted(losses_val)
        if (len(losses_val) >1) and (losses_val[-1] > losses_val[-2]):
            if it == 1:
                break
            else:
                it = 1
        else:
            torch.save(rnn.state_dict(),os.path.join(os.getcwd(),'model-'+str(epoch)))
            it = 0
            epoch += 1

else:
    print('Wrong option. 1 for RNN or 0 for FFNN')
'''
train = np.array(losses)
val = np.array(losses_val)
plt.plot(range(train.shape[0]), train.reshape(train.shape[0]), label='Training loss')
plt.plot(range(val.shape[0]), val.reshape(train.shape[0]), label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Cross-entropy loss of recurrent neural network')
plt.legend()

fpr_v, fnr_v, eer_v, acc_v = det_rnn(loader_val, rnn, device)
fpr_t, fnr_t, eer_t, acc_t = det_rnn(loader_test, rnn, device)

print('Val: EER: {}, ACC: {}'.format(eer_v, acc_v))
print('Test: EER: {}, ACC: {}'.format(eer_t, acc_t))

fpr_v = np.array(fpr_v)
fnr_v = np.array(fnr_v)
fpr_t = np.array(fpr_t)
fnr_t = np.array(fnr_t)

plt.plot(fpr_v,fnr_v, label = 'Validation')
plt.plot(fpr_t,fnr_t, label = 'Test')
plt.xlabel('FPR')
plt.ylabel('FNR')
plt.title('DET curve of recurrent neural network')
plt.legend()

loss_test = val_rnn(loader_test, rnn, device, criterion)
print('Test: Epoch {}: loss: {}'.format(epoch, loss_test.item()))

# Predict everything as 1
tr1 = labels_tr.sum()
trt = labels_tr.shape[0]
print(tr1/trt)

# Predict everything as 0
print((trt - tr1)/trt)

# Predict everything as 1
val1 = labels_val.sum()
valt = labels_val.shape[0]
print(val1/valt)

# Predict everything as 0
print((valt - val1)/valt)

# Predict everything as 1
test1 = labels_test.sum()
testt = labels_test.shape[0]
print(test1/testt)

# Predict everything as 0
print((testt - test1)/testt)
'''