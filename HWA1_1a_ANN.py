#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 12:50:22 2018

@author: MichaelBiehler
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('HWA1_1a_Dataset.csv')
train_X = dataset.iloc[:, [0, 1]].values
train_y = dataset.iloc[:, 2].values


import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

mlp = Sequential()
mlp.add(Dense(16, activation='relu', input_shape=(2,)))
mlp.add(Dense(16, activation='relu'))
mlp.add(Dense(16, activation='relu'))
mlp.add(Dense(16, activation='relu'))
mlp.add(Dense(16, activation='tanh'))
mlp.add(Dense(16, activation='tanh'))
mlp.add(Dense(16, activation='tanh'))
mlp.add(Dense(16, activation='tanh'))
mlp.add(Dense(16, activation='tanh'))
mlp.add(Dense(16, activation='tanh'))
mlp.add(Dense(16, activation='tanh'))
mlp.add(Dense(16, activation='relu'))
mlp.add(Dense(16, activation='relu'))
mlp.add(Dense(16, activation='relu'))
mlp.add(Dense(16, activation='relu'))
mlp.add(Dense(1, activation='sigmoid'))
mlp.compile(loss='binary_crossentropy', optimizer=RMSprop(),
              metrics=['accuracy'])
 mlpFit = mlp.fit(train_X, train_y, epochs=40000, verbose=0)
score = mlp.evaluate(train_X, train_y)
print(score)



# Visualising the classifying results
from matplotlib.colors import ListedColormap
X_set, y_set = X, y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, mlp.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('ANN Two Spirals Task 1a;40,000 epochs, more layers')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.show()
