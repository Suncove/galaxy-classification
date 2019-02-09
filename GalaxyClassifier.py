#! /usr/bin/env python3

<<<<<<< HEAD
#Copyright 2019 Kyle Steckler
=======
# Copyright 2019 Kyle Steckler
>>>>>>> 76cf2aac68482463da22774549283b6f763d1fdc

# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software 
# without restriction, including without limitation the rights to use, copy, modify, merge, 
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons 
# to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or 
# substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import confusion_matrix
import os
import pdb

def min_max_normalize(x):
    norm_arr = np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
    assert (norm_arr.max(), norm_arr.min()) == (1.0, 0.0), "Image not normalized properly. May contain NaN values"

    return norm_arr
    
def clean_data(images, labels):
    """
    PARAMS
    images: N images as arrays with dimensions:(N, H, W, channels)
    labels: Array of labels
    
    RETURNS
    normalized_images, encoded_labels
    """
    
    # Encode classification labels
    encoded_labels = to_categorical(labels)
    
    # normalize images
    normalized_images = np.array([min_max_normalize(x) for x in images])

    return normalized_images, encoded_labels

def get_data(image_data = 'galaxy_images.npy', galaxy_data = 'galaxy_data.csv', target='Classification'):
    image_data = np.load(image_data)
    galaxy_data = pd.read_csv(galaxy_data)
    labels = np.array(galaxy_data[target])
    return image_data, labels

def create_cnn(input_shape):
    model = Sequential()
    
    model.add(Conv2D(16, kernel_size=(2,2), padding = 'same',input_shape = input_shape, activation ="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    ## REUQIRES > 16GB RAM
    # model.add(Conv2D(32, (2, 2), padding='same', activation="relu"))
    # model.add(BatchNormalization(axis=-1))
    # model.add(MaxPooling2D(pool_size = (2,2)))
    # model.add(Dropout(0.25))

    # REQUIRES > 32GB RAM
    # model.add(Conv2D(64, (2, 2), padding='same', activation="relu"))
    # model.add(BatchNormalization(axis=-1))
    # model.add(MaxPooling2D(pool_size = (2,2)))
    # model.add(Dropout(0.25))  \
      
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(2, activation="softmax"))

    return model

def plot_loss(history):
    val_loss = history.history['val_loss']
    loss = history.history['loss']

    acc = history.history['acc']
    val_acc = history.history['val_acc']    
    
    plt.figure()
    
    plt.subplot(121)
    plt.title('Loss')
    plt.plot(val_loss, label = 'val_loss')
    plt.plot(loss, label = 'Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(122)
    plt.title('Accuracy')
    plt.plot(val_acc, label='val_acc')
    plt.plot(acc, label='training acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    # Load data
    image_data, labels = get_data()

    # Normalize images and encode labels
    X, y = clean_data(image_data, labels)

    # Train/Test split
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=42)
    
    # Create/Compile/Fit
    model = create_cnn(X_train[0].shape)
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
<<<<<<< HEAD
    performance = model.fit(X_train, y_train, batch_size=5, epochs = 5, validation_split=0.2, verbose=1)
=======
    history = model.fit(X_train, y_train, batch_size=5, epochs = 5, validation_split=0.2, verbose=1)

    # Grab performance specs
    val_loss = history.history['val_loss']
    loss = history.history['loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
>>>>>>> 76cf2aac68482463da22774549283b6f763d1fdc

    # predict on X_test, get confusion matrix
    predictions = model.predict(X_test)
    y_pred = [np.argmax(p) for p in predictions]
    y_true = [np.argmax(x) for x in y_test]
    conf_mat = confusion_matrix(y_true, y_pred)
<<<<<<< HEAD
    print(conf_mat)

    plot_loss(performance)



    model.save('my-galaxy-model.h5')
    
    K.clear_session()
=======
    print(conf_mat)    
    
    # plot loss
    plt.plot(val_loss, label = 'val_loss')
    plt.plot(loss, label = 'loss')
    plt.legend()
    plt.show()

    # plot accuracy
    plt.plot(val_acc, label='val_acc')
    plt.plot(acc, label='acc')
    plt.legend()
    plt.show()
    
    model.save('my-galaxy-model.h5')
    K.clear_session()
    
>>>>>>> 76cf2aac68482463da22774549283b6f763d1fdc


    

    
       
