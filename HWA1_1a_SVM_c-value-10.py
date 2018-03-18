#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 22:31:49 2018

@author: martinthoma

notes: first script to classify the two spirals with sklearn package. 
Best result with C = 10. For higher C-values no significant improvement

C : float, optional (default=1.0)
Penalty parameter C of the error term.
When noicy data, use C < 1. Because out data is clean we take high C value
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
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, C = 10)
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
plt.title('Classifier Two Spirals Task 1a (C = 10)')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.show()

