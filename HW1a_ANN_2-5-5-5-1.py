# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Michael Biehler
#adapted code from lab2
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
import csv
import tensorflow as tf
from numpy import arange, round, meshgrid, resize
import matplotlib.pyplot as plt

def read_two_spiral_file(filename="/Users/oem/Documents/machine learning/Data.csv"):
    x = []
    y = []
    
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            x.append(list(map(float, row[:-1])))
            y.append([int(row[-1])])

    return x, y

x, y = read_two_spiral_file()


# Create the model
# Lang and Witbrock (1988) 2-5-5-5-1 MLP
x_ = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 1])

# Create first layer weights
layer_0_weights = tf.Variable(tf.random_normal([2, 5]))
layer_0_bias = tf.Variable(tf.random_normal([5]))
layer_0 = tf.nn.sigmoid(tf.add((tf.matmul(x_, layer_0_weights)), layer_0_bias))

# Create second layer weights
layer_1_weights = tf.Variable(tf.random_normal([5, 5]))
layer_1_bias = tf.Variable(tf.random_normal([5]))
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(layer_0, layer_1_weights), layer_1_bias))

# Create third layer weights
layer_2_weights = tf.Variable(tf.random_normal([5, 5]))
layer_2_bias = tf.Variable(tf.random_normal([5]))
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, layer_2_weights), layer_2_bias))

# Create fourth layer weights
layer_3_weights = tf.Variable(tf.random_normal([5, 5]))
layer_3_bias = tf.Variable(tf.random_normal([5]))
layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, layer_3_weights), layer_3_bias))


# Create fifth layer weights
layer_4_weights = tf.Variable(tf.random_normal([5, 1]))
layer_4_bias = tf.Variable(tf.random_normal([1]))
layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, layer_4_weights), layer_4_bias))


# Define error function
cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=layer_4))

# Define optimizer and its task (minimise error function)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.5).minimize(cost)

N_EPOCHS = 10000  #like in paper my Chalup&wiklent 

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

print('Training...')

errors = []

# Train
for i in range(N_EPOCHS):
    _, error = sess.run([optimizer,cost], feed_dict={x_: x, y_: y})
    errors.append(error)
    
plt.plot(errors)
plt.show()     

# Visualise activations
activation_range = arange(-6,6,0.1) # interval of [-6,6) with step size 0.1
coordinates = [(x,y) for x in activation_range for y in activation_range]
classifications = round(sess.run(layer_4, feed_dict={x_:coordinates}))
x, y = meshgrid(activation_range, activation_range)
plt.scatter(x, y, c=['b' if x > 0 else 'y' for x in classifications])
plt.show()