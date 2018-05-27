#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 19:45:42 2018

@author: matthiasboeker
"""

import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy
import pandas as pd
from scipy.misc import toimage

data_x = np.load('X.npy')
data_y = np.load('Y.npy')


def split_data(data, index):

    size = data.shape[0] 
    test_data = list()
    train_data = list()
    k = 0
    for i in range(0,(size-1)//index):
        temp = data[i*index]
        test_data.append(temp)
        if i != 0:
            while k < i*index:
                 tempo = data[k]
                 train_data.append(tempo)  
                 k = k+1
        k = k + 1
    test_array = np.asarray(test_data)  
    train_array = np.asarray(train_data)        
    return test_array, train_array

