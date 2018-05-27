# -*- coding: utf-8 -*-
"""
Created on Sun May  6 18:11:21 2018

@author: Fiona Mallett
Student Number: 3289339
"""

from keras.models import Sequential  #used to initialise CNN
from keras.layers import Convolution2D #adding conoltuion layers
from keras.layers import MaxPooling2D # add pooling layers
from keras.layers import Flatten #flattening
from keras.layers import Dense #add fully connected layers
from keras.layers import Dropout #prevent overfitting


import numpy as np

data_x = np.load('X.npy')
data_y = np.load('Y.npy')


x_test, x_train = split_data(data_x, 4)
y_test, y_train = split_data(data_y, 4)


#Initialising the CNN
classifier  = Sequential()

#Step 1: Convolution - feature dector - find features in image
 #paramteters of convolution layer.. number of filters, rows and columns
classifier.add(Convolution2D(64, 3, 3, input_shape=(50, 50, 3), activation = 'relu'))  #good practice to start with 32, then 64, 128..
classifier.add(Convolution2D(64, 3, 3, input_shape=(50, 50, 3), activation = 'relu'))  
classifier.add(Convolution2D(64, 3, 3, input_shape=(50, 50, 3), activation = 'relu'))

#Step 2: Max Pooling - stride of 2
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3: Flattening -into 1D vector
classifier.add(Flatten())

#Step 4: Full Connection
#add hidden layer
classifier.add(Dense(output_dim = 128, activation = 'relu')) #no. of nodes in hidden layer (between no of nodes in input and output)

#Add dropout layer
#keras.layers.Dropout(rate, noise_shape=None, seed=None)
classifier.add(Dropout(0.4))

#output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) #sigmoid for binary output

#Compile the CNN 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Convert numpy array to a folder containing the images
convert_to_image(x_train, "training", y_train)
convert_to_image(x_test, "testing", y_test)


#-----Not sure about keeping this as we already have images of the same size and in folders... 
#I think all we need to do is fit the classifier with the input arrays to train

#Fit CNN to the input -- from keras documentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory('dataset/training', #this will be a folder in our directory
                                                target_size=(50, 50),
                                                batch_size=32,
                                                class_mode='binary') #2 classes is binary eg cat and dog

test_set = test_datagen.flow_from_directory('dataset/testing',
                                            target_size=(50, 50),
                                            batch_size=32,
                                            class_mode='binary')

CNN = classifier.fit_generator(training_set,
                    steps_per_epoch=20, #no. of images in training set
                    epochs=500,
                    validation_data=test_set,
                    validation_steps=10) #no. of images in test set


#Used to output data in graph form
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6,5)
plt.plot(CNN.history['acc'])
plt.plot(CNN.history['val_acc'])
plt.title( "Accuracy ")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.close()
# summarize history for loss
plt.plot(CNN.history['loss'])
plt.plot(CNN.history['val_loss'])
plt.title("Error")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
plt.close()