#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 21:18:41 2018

@author: MichaelBiehler
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('h1b_data.csv')


# format data 

noWithdrawnData = dataset[dataset['CASE_STATUS']!='WITHDRAWN']
twoClassData    = noWithdrawnData[noWithdrawnData['CASE_STATUS']!='CERTIFIED-WITHDRAWN']
 

#ONE-HOT Encoding of 
dummyData = pd.get_dummies(twoClassData.drop('CASE_STATUS', axis = 1))
mergedData = pd.concat([dummyData, twoClassData['CASE_STATUS']], axis=1)
formatedData = mergedData.dropna() #dataset with all categories encoded exept of dependent y

#defining independent variables (X) and dependent (y)
from sklearn.cross_validation import train_test_split

X = formatedData.drop('CASE_STATUS', axis = 1)  
y = formatedData['CASE_STATUS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# haveing a first view into the data
X_train.head()
y_train.head()
y_test.head()
X_test.head()




'''
# Taking care of missing data (deleting any patterns with "nan")
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(dataset[:, [8, 9]])
dataset[:, [8, 9]] = imputer.transform(dataset[:, [8, 9]])
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



#code y_train and y_test to numbers (certified = 0, denied = 1)
y_train_codes = y_train.astype('category').cat.codes
y_train=y_train_codes
y_test_codes = y_test.astype('category').cat.codes
y_test=y_test_codes

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13,13,13,13,13),max_iter=500,
                    activation='logistic')
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,roc_curve
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# calculate the fpr and tpr for all thresholds of the classification
from sklearn import metrics
probs = mlp.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test_codes, predictions)
roc_auc = metrics.auc(fpr, tpr)

# plot ROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.plot([0, 0], [0, 1], c = 'black')
plt.plot([0, 1], [1, 1], c = 'black')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#evaluation of the SVM via confusion matrix and parameters darivated from confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_codes, predictions)

tpr = cm[0, 0]/(cm[0, 0]+cm[1, 0]) #true positive rate, sensitivity, recall
tnr = cm[1, 1]/(cm[1, 1]+cm[0, 1]) #true negative rate, specificy
acc = (cm[1, 1] + cm[0, 0])/(cm[1, 1] + cm[0, 1] + cm[1, 0] + cm[0, 0])


#evaluation of the SVM via classification_report of sklearn
#recall of positive class == sensitivity ;; recall of negative class == specificity
from sklearn.metrics import classification_report
print(classification_report(y_test_codes, predictions))

# create CAP Curve 

from scipy import integrate

y_values = y_test_codes
y_preds_proba = preds

num_pos_obs = np.sum(y_values)
num_count = len(y_values)
rate_pos_obs = float(num_pos_obs) / float(num_count)
ideal = pd.DataFrame({'x':[0,rate_pos_obs,1],'y':[0,1,1]})
xx = np.arange(num_count) / float(num_count - 1)

y_cap = np.c_[y_values,y_preds_proba]
y_cap_df_s = pd.DataFrame(data=y_cap)
y_cap_df_s = y_cap_df_s.sort_values([1], ascending=False)
y_cap_df_s = y_cap_df_s.reset_index(drop=True)
print(y_cap_df_s.head(20))

yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)
yy = np.append([0], yy[0:num_count-1]) #add the first curve point (0,0) : for xx=0 we have yy=0

percent = 0.5
row_index = int(np.trunc(num_count * percent))

val_y1 = yy[row_index]
val_y2 = yy[row_index+1]
if val_y1 == val_y2:
   val = val_y1*1.0
else:
    val_x1 = xx[row_index]
    val_x2 = xx[row_index+1]
    val = val_y1 + ((val_x2 - percent)/(val_x2 - val_x1))*(val_y2 - val_y1)

sigma_ideal = 1 * xx[num_pos_obs - 1 ] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1
sigma_model = integrate.simps(yy,xx)
sigma_random = integrate.simps(xx,xx)

ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random)
#ar_label = 'ar value = %s' % ar_value

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.plot(ideal['x'],ideal['y'], color='grey', label='Perfect Model')
ax.plot(xx,yy, color='red', label='ANN Model')
#ax.scatter(xx,yy, color='red')
ax.plot(xx,xx, color='blue', label='Random Model')
ax.plot([percent, percent], [0.0, val], color='green', linestyle='--', linewidth=1)
ax.plot([0, percent], [val, val], color='green', linestyle='--', linewidth=1, label=str(val*100)+'% of positive obs at '+str(percent*100)+'%')

plt.xlim(0, 1.02)
plt.ylim(0, 1.25)
plt.title("CAP Curve - a_r value ="+str(ar_value))
plt.xlabel('% of the data')
plt.ylabel('% of positive obs')
plt.legend()
plt.show()


import pickle
pickle.dump(clf,open('fileout','wb'))







