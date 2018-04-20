#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 22:31:49 2018

@author: martinthoma

"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data_h1b.csv')


# format data 
'''
Want just to figure out if visa is granted or denied wo we dont need withdrawn visa application 
Removing 'Withdrawn' field
Merge 'Certified-Withdrawn' and 'Certified' 
encoding all classified data


'''
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# haveing a first view into the data
print(X.describe())
print(y.describe())

'''
# Taking care of missing data (deleting any patterns with "nan")
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(dataset[:, [8, 9]])
dataset[:, [8, 9]] = imputer.transform(dataset[:, [8, 9]])
'''

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#code y_train and y_test to numbers (certified = 0, denied = 1)
y_train_codes = y_train.astype('category').cat.codes
y_test_codes = y_test.astype('category').cat.codes
y_codes = y.astype('category').cat.codes   


# Fitting classifier to the Training set
#random_state parameter in SVC function is set to 0 to have the same results in group
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, C = 1, probability = True)
classifier.fit(X_train, y_train_codes)

#predict results via classifier
y_pred = classifier.predict(X_test)


# calculate the fpr and tpr for all thresholds of the classification
from sklearn import metrics
probs = classifier.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test_codes, preds)
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
cm = confusion_matrix(y_test_codes, y_pred)

tpr = cm[0, 0]/(cm[0, 0]+cm[1, 0]) #true positive rate, sensitivity, recall
tnr = cm[1, 1]/(cm[1, 1]+cm[0, 1]) #true negative rate, specificy
acc = (cm[1, 1] + cm[0, 0])/(cm[1, 1] + cm[0, 1] + cm[1, 0] + cm[0, 0])


#evaluation of the SVM via classification_report of sklearn
#recall of positive class == sensitivity ;; recall of negative class == specificity
from sklearn.metrics import classification_report
print(classification_report(y_test_codes, y_pred))

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
ax.plot(xx,yy, color='red', label='SVM Model')
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



'''

print(__doc__)

from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# #############################################################################

# ROC analysis
cv = StratifiedKFold(n_splits=6)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

train_indices, test_indices = cv.split(X, y)

i = 0
for train in X_train:
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Visualising the classifying results for trainingsset
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train_codes
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier Two Spirals Task 1a (C = 1)')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.show()

# Visualising the classifying results for test set
# takeing same scale like for the test set
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_pred
X1, X2 = np.meshgrid(np.arange(start = X_train[:, 0].min() - 1, stop = X_train[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_train[:, 1].min() - 1, stop = X_train[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier Two Spirals Task 1a (C = 1)')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.show()

'''
