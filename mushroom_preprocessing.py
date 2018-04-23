#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 10:19:29 2018

@author: matthiasboeker
"""

#Data preprocessing function

    #Only appliable for mushroom data
    #Transformes categorical data by standardization or transformation to dummy variables 
    #Input: data, switch: (Choose between standardization (0) or dummy variables (1)), test_size, random_state
    #Output:  X_train, X_test, y_train, y_test

import pandas as pd 
import numpy as np
 
def data_preprocessing(data, switch = 0 ,test_size = 0.33, random_state = None):
    from sklearn.model_selection import train_test_split
    
    if switch == 0:  
       from sklearn import preprocessing
       label_encoder =preprocessing.LabelEncoder()
       
       X = data.iloc[:,1:23]
       y = data.iloc[:,0]

       
       for i in X.columns:
           X[i] = label_encoder.fit_transform(X[i])
                   
       scaler = preprocessing.StandardScaler()
       X =scaler.fit_transform(X)
      
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state )
       return(X_train, X_test, y_train, y_test,X ,y)
            
    elif switch == 1:
         X = data.iloc[:,1:23]
         y = data.iloc[:,0]
         
         
         y = pd.get_dummies(y)
         y = y.iloc[:,1:].values
         #loop through the rows 
         X_afterDummyTrap = pd.DataFrame()
         for k in X.columns:
             X_dummy = pd.get_dummies(X[k])
             X_dummy = X_dummy.iloc[:,1:].values
             X_dummy = pd.DataFrame(X_dummy)
             X_afterDummyTrap = pd.concat([X_afterDummyTrap,X_dummy] , axis=1)
             
         
         X_train, X_test, y_train, y_test = train_test_split(X_afterDummyTrap, y, test_size = test_size, random_state = random_state)
         return(X_train, X_test, y_train, y_test, X_afterDummyTrap ,y )
    else:
        print('ERROR: Input of switch variable must be 0 or 1')
    