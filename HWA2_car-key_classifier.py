#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 13:47:50 2018

@author: martinthoma
"""

# Convolutional Neural Network for Image Classification

'''
Requirements: 
    
 Installing Theano
 pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

 Installing Tensorflow
 Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

 Installing Keras
 pip install --upgrade keras
'''


# Because of folderstructure we dont have to do data preprocessing anymore
# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D #use to deal with images
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
#create 32 feature detectors (start with 32, double later in further convolutional layers cause we run on CPU)
#input_shape: force images in a shape. B&W Pictures need 3 in the end (cause tensorflow backend). 
#relu function will get non-linearity. Removes negative pixels
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
#no input_shape needed because network gets that from previous pooling step
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
#How many nodes in the hidden layer? Choose a number between number of input and output nodes
#mostly use a power of 2
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images (Image Preprocessing also making image set larger)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/images/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000, 
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)

#to optimize such models 1. add another convolutional + pooling layer
# 2. add another ANN layer