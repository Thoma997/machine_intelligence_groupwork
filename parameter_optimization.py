#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 11:27:58 2018

@author: matthiasboeker
"""

import pandas as pd 
import numpy as np
 
from sklearn import svm
Support_Vector_Machine_Obj = svm.SVC()
n_iter_search = 5

def para_opti(Support_Vector_Machine_Obj, X_train, y_train, X_test, y_test, parameters, case = 0, n_iter_search = 1):
    from sklearn.metrics import classification_report
    
    if case == 0:
        #Run randomized search
        from sklearn.model_selection import RandomizedSearchCV  
        random_search = RandomizedSearchCV(Support_Vector_Machine_Obj, param_distributions=parameters, n_iter=n_iter_search)
        random_search.fit(X_train,y_train)
        print("Best parameters:", random_search.best_params_)
        print(classification_report(y_test, random_search.predict(X_test)))
        return(random_search.best_params_)
    elif case == 1:
        #Grid Search Method
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(Support_Vector_Machine_Obj, parameters)
        grid_search.fit(X_train, y_train)
        print("Best parameters:", grid_search.best_params_)
        print(classification_report(y_test, grid_search.predict(X_test))) 
        return(grid_search.best_params_)
    else:
        print('ERROR: Input of case variable must be 0 or 1')



    