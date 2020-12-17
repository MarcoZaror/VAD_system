# -*- coding: utf-8 -*-
"""
Created on Tue May 12 08:27:27 2020

@author: Marco Zaror
File created to evaluate a feed-forward neural network or a recurrent neural 
network. Receive as input a irectory containing the audio files, 
another containing the labels and a flag indicating to evaluate the FFNN
 or RNN.

USE: python <PROGNAME> (options) 
    - directory audio
    - directory labels
    - 1 for RNN or 0 for FFNN
    - name of the model
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from train_eval_func import pred, compute_eer, train_ffnn, val_ffnn, eer_acc_ffnn, train_rnn, val_rnn, eer_acc_rnn, det_ffnn, det_rnn
from Data import loadData, addContext, preprocessData, preprocessSeqData
from model import FFN, RNN
import torch
from Data2 import Data2
import torch.nn as nn
from pathlib import Path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path_audio = Path(sys.argv[1])
path_labels = Path(sys.argv[2])

# Load data
#files_val = loadData(path_audio,['EDI'])
#labels_val = loadData(path_labels,['EDI']).astype(float)
#files_test = loadData(path_audio,['CMU'])
#labels_test = loadData(path_labels,['CMU']).astype(float)
data_val = Data2()
data_test = Data2()
data_val.loadData(path_audio,['EDI'])
data_val.load_labels(path_labels,['EDI']).astype(float)
data_test.loadData(path_audio,['CMU'])
data_test.load_labels(path_labels,['CMU']).astype(float) 
print('Data loaded')

if sys.argv[3] == '0':
    # Preprocess data
    data_val.addContext(2)
    data_test.addContext(2)
    loader_test,_ = data_val.preprocessData(subsample=0, bs=64)
    loader_val,_ = data_test.preprocessData(subsample=0, bs=64)
    print('Data processed')
    
    ffnn = FFN(65, 150, 2)
    ffnn.load_state_dict(torch.load(sys.argv[4]))
    
    fpr_v, fnr_v, eer_v, acc_v = det_ffnn(loader_val, ffnn, device)
    fpr_t, fnr_t, eer_t, acc_t = det_ffnn(loader_test, ffnn, device)
    
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
    plt.title('DET curve of feed-forward neural network')
    plt.legend()
    plt.show()
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    loss_test = val_ffnn(loader_test, ffnn, device, criterion)
    print('Test: Loss: {}'.format(loss_test.item()))
    
    ''' Naive classifier
    # Predict everything as 1
    test1 = labels_test.sum()
    testt = labels_test.shape[0]
    print(test1/testt)
    
    # Predict everything as 0
    print((testt - test1)/testt)
    '''
elif sys.argv[3] == '1':
        # Preprocess data
    loader_test,_ = data_val.preprocessSeqData(subsample=0.2,seq_len=3, bs=64)
    loader_val,_ = data_test.preprocessSeqData(subsample=0.2,seq_len=3, bs=64)
    print('Data processed')
    
    rnn = RNN(13, 100, 2, 1)
    rnn.load_state_dict(torch.load(sys.argv[4]))
    
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
    plt.title('DET curve of feed-forward neural network')
    plt.legend()
    plt.show()
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    loss_test = val_rnn(loader_test, rnn, device, criterion)
    print('Test: Loss: {}'.format(loss_test.item()))

    ''' Naive classifier
    # Predict everything as 1
    test1 = labels_test.sum()
    testt = labels_test.shape[0]
    print(test1/testt)
    
    # Predict everything as 0
    print((testt - test1)/testt)
    '''
else:
    print('Wrong option. 1 for RNN or 0 for FFNN')