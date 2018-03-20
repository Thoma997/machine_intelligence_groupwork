#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 22:31:49 2018

@author: martinthoma

notes: 
KNN algorythmus is calculating the K nearest neighbors (with euclid. distance).
Assigns the point to the class to which the majority of the nearest points belongs to.

You can change parameter "n_neighbors" to values 1,3,5,7 to see the impact of amount of neighbors choosen.
please use only uneven numbers cause otherwise the algo cant assign points to a class.
In this case you will get the "nicest" spiral by choosing "n_neighbors = 1"

"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('HWA1_1a_Dataset.csv')
X = dataset.iloc[:, [0, 1]].values
y = dataset.iloc[:, 2].values

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)"""

# Fitting classifier to the Training set
#random_state parameter in SVC function is set to 0 to have the same results in group
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier( n_neighbors = 1, metric = 'minkowski', p = 2 )
classifier.fit(X, y)

# Visualising the classifying results
from matplotlib.colors import ListedColormap
X_set, y_set = X, y
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

