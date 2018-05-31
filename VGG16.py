#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:32:09 2018

@author: matthiasboeker
VGG16 Model
"""

from keras.models import Sequential  #used to initialise CNN
from keras.callbacks import History 
history = History()


import numpy as np

data_x = np.load('X.npy')
data_y = np.load('Y.npy')

weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels.h5"

data_x = resize_images(data_x)

x_test, x_train = split_data(data_x, 4)
y_test, y_train = split_data(data_y, 4)


#Image preprocessing



from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from sklearn.metrics import log_loss


def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):
    """VGG 16 Model for Keras
    Model Schema is based on 
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    ImageNet Pretrained Weights 
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of categories for our classification task
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=( img_rows, img_cols,channel)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='sigmoid'))

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    #for layer in model.layers[:10]:
    #    layer.trainable = False

    # Learning rate is changed to 0.001
    adam = Adam(lr=0.0001)
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model




    
img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
num_classes = 1 
batch_size = 16#33
nb_epoch = 20

# Load our model
model = vgg16_model(img_rows, img_cols, channel, num_classes)

# Start Fine-tuning
VGG16_cnn = model.fit(x_train[1968:2168,:,:,:], y_train[1968:2168],
                      batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      shuffle=True,
                      verbose=1,
                      validation_data=(x_test[670:710,:,:,:], y_test[670:710]))

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6,5)
plt.plot(VGG16_cnn.history['acc'])
plt.plot(VGG16_cnn.history['val_acc'])
plt.title( "Accuracy ")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.close()
# summarize history for loss
plt.plot(VGG16_cnn.history['loss'])
plt.plot(VGG16_cnn.history['val_loss'])
plt.title("Error")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
plt.close()