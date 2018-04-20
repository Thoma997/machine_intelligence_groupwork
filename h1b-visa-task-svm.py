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

#code y_train and y_test to numbers
y_train_codes = y_train.astype('category').cat.codes #(certified = 0, denied = 1)
y_test_codes = y_test.astype('category').cat.codes   #(certified = 0, denied = 1)


# Fitting classifier to the Training set
#random_state parameter in SVC function is set to 0 to have the same results in group
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, C = 1)
classifier.fit(X_train, y_train_codes)

#predict results via classifier
y_pred = classifier.predict(X_test)


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



import mpl_toolkits

f1, ax1 = plt.subplots(1)
    lons = X['lon'].values
    lats = X['lat'].values
    # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
    # are the lat/lon values of the lower left and upper right corners
    # of the map.
    # lat_ts is the latitude of true scale.
    # resolution = 'c' means use crude resolution coastlines.
    m = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=55,
            llcrnrlon=-135,urcrnrlon=-60,lat_ts=20,resolution='l')
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    #m.drawmapboundary(fill_color='aqua')
    x,y = m(lons,lats)
    m.scatter(x,y,marker='o')
    plt.title("Mercator Projection of H-1B Visa Destinations")
    
    
    
    f3, ax3 = plt.subplots(1)
    data.hist('PREVAILING_WAGE', bins = 1000, ax=ax3)
    ax3.set_xlim([0,150000])
    ax3.set_xlabel('Wage ($)')
    ax3.set_ylabel('Number of Applicants')

    plt.show()
    


'''
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
