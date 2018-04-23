#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:35:17 2018

@author: matthiasboeker
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import time

data = pd.read_csv('mushrooms.csv')

#Data preprocessing - Standardize
X_train_s, X_test_s, y_train_s, y_test_s, X_s,y_s = data_preprocessing(data, 0,  0.5, random_state = 4)

#Data preprocessing - Dummy variables
X_train_d, X_test_d, y_train_d, y_test_d, X_d,y_d = data_preprocessing(data, 1,  0.5, random_state = 4)


#Fitting to SVC model
from sklearn import svm
cl = svm.SVC()

#Optimizing the parameters of the SVM
#Spezifiy parameters and distributions to sample
parameters = {'kernel': ('linear', 'rbf'),
                'C':[ 0.001, 0.01, 0.1, 1],
                'gamma':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
              }
#Optimizing parameters - Dummy - Grid
start = time.time()
best_para_d_G = para_opti(cl, X_train_d, y_train_d, X_test_d, y_test_d, parameters, 1 , 10)
end = time.time()
print('Dummy-Grid Search Time',end - start, ' secs')
#Optimizing parameters - Dummy - Random
start = time.time()
best_para_d_R = para_opti(cl, X_train_d, y_train_d, X_test_d, y_test_d, parameters, 0 , 10)
end = time.time()
print('Dummy-Random Search Time',end - start, ' secs')
#Optimizing parameters - Stand - Grid
start = time.time()
best_para_s_G = para_opti(cl, X_train_s, y_train_s, X_test_s, y_test_s, parameters, 1 , 10)
end = time.time()
print('Standardized-Grid Search Time',end - start, ' secs')
#Optimizing parameters - Stand - Random
start = time.time()
best_para_s_R = para_opti(cl, X_train_s, y_train_s, X_test_s, y_test_s, parameters, 0 , 10)  
end = time.time()
print('Standardized-Random Search Time', end - start, ' secs')

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
  
#Fitting to SVC model with optimized parameters
clf_d_G = svm.SVC(best_para_d_G['C'],best_para_d_G['kernel'],3,best_para_d_G['gamma'], probability = True)
clf_d_G.fit(X_train_d,y_train_d)
#Fitting test data
y_predict_d_G = clf_d_G.predict(X_test_d)

score = accuracy_score(y_test_d, y_predict_d_G)
cm = confusion_matrix(y_test_d, y_predict_d_G)
print('Dummy-Grid Score' ,score)
print('Dummy-Grid Confusion matrix', cm )

#Fitting to SVC model with optimized parameters
clf_d_R = svm.SVC(best_para_d_R['C'],best_para_d_R['kernel'],3,best_para_d_R['gamma'], probability = True)
clf_d_R.fit(X_train_d,y_train_d)
#Fitting test data
y_predict_d_R = clf_d_R.predict(X_test_d)

score = accuracy_score(y_test_d, y_predict_d_R)
cm = confusion_matrix(y_test_d, y_predict_d_R)
print('Dummy-Random Score' , score)
print('Dummy-Random Confusion matrix' , cm)

#Fitting to SVC model with optimized parameters
clf_s_G = svm.SVC(best_para_s_G['C'],best_para_s_G['kernel'],3,best_para_s_G['gamma'], probability = True)
clf_s_G.fit(X_train_s,y_train_s)
#Fitting test data
y_predict_s_G = clf_s_G.predict(X_test_s)

score = accuracy_score(y_test_s, y_predict_s_G)
cm = confusion_matrix(y_test_s, y_predict_s_G)
print('Standardize-Grid Score' , score)
print('Standardize-Grid Confusion matrix' , cm)

#Fitting to SVC model with optimized parameters
clf_s_R = svm.SVC(best_para_s_R['C'],best_para_s_R['kernel'],3,best_para_s_R['gamma'], probability = True)
clf_s_R.fit(X_train_s,y_train_s)
#Fitting test data
y_predict_s_R = clf_s_R.predict(X_test_s)

score = accuracy_score(y_test_s, y_predict_s_R)
cm = confusion_matrix(y_test_s, y_predict_s_R)
print('Standardize-Random Score' , score)
print('Standardize-Random Confusion matrix' , cm)

# calculate the fpr and tpr for all thresholds of the classification
from sklearn import metrics
probs = clf_d_G.predict_proba(X_test_d)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test_d, preds)
roc_auc = metrics.auc(fpr, tpr)



# plot ROC curve
plt.title('Dummy Variables Data - Random-Search Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.plot([0, 0], [0, 1], c = 'grey')
plt.plot([0, 1], [1, 1], c = 'grey')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


probs = clf_d_R.predict_proba(X_test_d)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test_d, preds)
roc_auc = metrics.auc(fpr, tpr)

# plot ROC curve
plt.title('Dummy Variables Data - Grid-Search Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.plot([0, 0], [0, 1], c = 'grey')
plt.plot([0, 1], [1, 1], c = 'grey')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


probs = clf_s_G.predict_proba(X_test_s)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test_d, preds)
roc_auc = metrics.auc(fpr, tpr)

# plot ROC curve
plt.title('Standardized Data - Grid-Search Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.plot([0, 0], [0, 1], c = 'grey')
plt.plot([0, 1], [1, 1], c = 'grey')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


probs = clf_s_R.predict_proba(X_test_s)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test_d, preds)
roc_auc = metrics.auc(fpr, tpr)

# plot ROC curve
plt.title('Standardized Data - Random-Search Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.plot([0, 0], [0, 1], c = 'grey')
plt.plot([0, 1], [1, 1], c = 'grey')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=4) #shuffle and split into n subsets

#k-fold Cross-Validation
scores = cross_val_score(clf_d_G, X_d ,y_s, cv=10).mean()
print('Dummy-Grid:',scores)
plot_learning_curve(clf_s_G, "Cross-validation on Dummy Variables Data - Grid-Search ", X_d, y_d, (0.7, 1.01), cv = cv, n_jobs=4)

#k-fold Cross-Validation
scores = cross_val_score(clf_d_R, X_d ,y_s, cv=10).mean()
print('Dummy-Random:',scores)
plot_learning_curve(clf_d_R, "Cross-validation on Dummy Variables Data - Random-Search ", X_d, y_d, (0.7, 1.01), cv = cv, n_jobs=4)

#k-fold Cross-Validation
scores = cross_val_score(clf_s_G, X_s ,y_s, cv=10).mean()
print('Standardize-Grid:',scores)
plot_learning_curve(clf_s_G, "Cross-validation on Standardized Variables Data - Grid-Search ", X_s, y_s, (0.7, 1.01),cv = cv,  n_jobs=4)

#k-fold Cross-Validation
scores = cross_val_score(clf_s_R, X_s ,y_s, cv=10).mean()
print('Standardize-Random:',scores)
plot_learning_curve(clf_s_R, "Cross-validation on Standardized Variables Data - Random-Search ", X_s, y_s,(0.7, 1.01), cv = cv, n_jobs=4)







