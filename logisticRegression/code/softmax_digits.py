# Main functionality of oftmax regression for MINST data
# training is done in ipython notebook with the same name because of 
# better visual environment

import resource

rsrc = resource.RLIMIT_DATA
soft, hard = resource.getrlimit(rsrc)
print 'Soft limit starts as  :', soft

resource.setrlimit(rsrc, (2024000000, hard)) #limit to one kilobyte

soft, hard = resource.getrlimit(rsrc)
print 'Soft limit changed to :', soft


import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nn.base import NNBase
from nn.math import softmax, make_onehot
from misc import random_weight_matrix
from numpy import *
from softmax_example import SoftmaxRegression

sns.set(color_codes=True)
plt.rcParams['savefig.dpi'] = 100


def get_data(num_train, num_dev, num_test):
    """
    read, preprocess and split the data into different sets. data has 
    following features pixel0 .. pixel783 (28 x 28 grayscale image)
    each has a value 0..255 inclusive, total number of images in train set are 
    4200.
        
        - num_train number of rows used for training
        - num_dev number of rows for validation
        - num_test number of rows for test split
    """
    fulltrain = pd.read_csv("../input/train.csv")
    fulltest = pd.read_csv("../input/test.csv") # we'd like a prediction for this

    # It seems that digits aren't uniformly distributed in the data
    # let's shuffle the whole training set to get a better splits
    #import ipdb; ipdb.set_trace() 
    ind = range(fulltrain.shape[0])
    np.random.shuffle(ind)
    fulltrainX = fulltrain.ix[ind].drop('label', axis=1).as_matrix()
    fulltrainy = fulltrain.ix[ind]['label'].as_matrix()
    fulltestX = fulltest.as_matrix()

    # normalize the data, substract mean and divide by SD
    mean = np.mean(fulltrainX, axis=0)
    sd = np.std(fulltrainX, axis=0)
    nonzero = sd > 0
    
    fulltrainX = fulltrainX - mean
    fulltrainX[:, nonzero] /= sd[nonzero]

    fulltestX = fulltestX - mean
    fulltestX[:, nonzero] /= sd[nonzero]

    # split the data 
    X_train = fulltrainX[:num_train, :]
    y_train = fulltrainy[:num_train]
    print 'X_train shape', X_train.shape

    X_dev = fulltrainX[num_train:num_train + num_dev,:]
    y_dev = fulltrainy[num_train:num_train + num_dev]
    
    print 'X_dev shape', X_dev.shape

    X_test = fulltrainX[num_train + num_dev:num_train + num_dev + num_test,:]
    y_test = fulltrainy[num_train+ num_dev:num_train + num_dev + num_test]
    
    print 'X_test shape', X_test.shape
    return (X_train, y_train, X_dev, y_dev, X_test, y_test,fulltestX)



#(X_train, y_train, X_dev, y_dev, X_test, y_test, X_fulltest) = get_data(num_train=40000, 
#        num_dev=1000, num_test=1000)
#sr = SoftmaxRegression(wv=zeros((10,784)), dims=(784,5))
#sr.grad_check(x=np.zeros(784), y=4)
#sr.predict_proba(np.zeros((2, 784)))
#sr = SoftmaxRegression(wv=zeros_like(X_train), dims=(784, 10))
#curve = sr.train_sgd(X_train, y_train, costevery=100, devX=X_dev, devy=y_dev)
