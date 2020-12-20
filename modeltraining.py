#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainingsdata generation script. Basic script structure and idea
by Adrain Rosebrock (https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/) 
@author: Florian Alkofer
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

from tensorflow.keras.applications import NASNetLarge, Xception

import numpy as np
import cv2
import os
import pickle



def model_NasNet(width, height, epochs, learnrate, numclasses):
    """
    Generate and train a NASNetLarge model using prepared data for a character recognition task.

    Parameters
    ----------
    width : int
        Input width.
    height : int
        Input height.
    epochs : int
        Number of epochs the model will be trained for.
    learnrate : float
        Learnrate for the model.
    numclasses : int
        Number of different classes in the training data set.

    Returns
    -------
    model : keras.model
        Tensorflow Keras model for Deep Learning applications. Trained with character data.

    """
    base_model = NASNetLarge(weights='imagenet', include_top=True, input_shape=(width,height,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(numclasses, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    opt = SGD(lr=learnrate, momentum=0.9, decay=learnrate/epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


def model_Xception(width, height, epochs, learnrate, numclasses):
    """
    Generate and train a NASNetLarge model using prepared data for a character recognition task.

    Parameters
    ----------
    width : int
        Input width.
    height : int
        Input height.
    epochs : int
        Number of epochs the model will be trained for.
    learnrate : float
        Learnrate for the model.
    numclasses : int
        Number of different classes in the training data set.

    Returns
    -------
    model : keras.model
        Tensorflow Keras model for Deep Learning applications. Trained with character data.

    """
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(width,height,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(numclasses, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    opt = SGD(lr=learnrate, momentum=0.9, decay=learnrate/epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


def genLencoder(architecture, labelfile = "trainData/labels.txt"):
    """
    Generate a labelencoder to de- and encode labels for a given label file.

    Parameters
    ----------
    labelfile : string
        Path to file containing label data for a set of training data
    architecture : string
        Name of the architecture the encoder will be saved under

    Returns
    -------
    labelEnc : sklearn.preprocessing.Label_Encoder
        Label Encoder used for initial encoding of the labels and for decoding in the output stage.

    """
    labels = []
    tr_labelsf = open(labelfile, "r")
    tr_labels = tr_labelsf.readlines()
    tr_labelsf.close()
    for i in tr_labels:
        labels.append(i[:-1])
    labelEnc = LabelEncoder()
    labels = labelEnc.fit_transform(labels)
    with open("encoders/" + architecture + ".sav", 'wb') as f:
        pickle.dump(labelEnc, f)
    return labelEnc


def train_model(width, height, epochs = 50, learnrate = 1e-1, numclasses = 77, labelfile = "trainData/labels.txt", architecture = "NASNet", trainData = "trainData", batchsize = 32, custom_name = "Xception"):
    """
    Generate and train a model using prepared data for a character recognition task.

    Parameters
    ----------
    width : int
        Input width.
    height : int
        Input height.
    epochs : int
        Number of epochs the model will be trained for.
    learnrate : float
        Learnrate for the model.
    numclasses : int
        Number of different classes in the training data set.
    labelfile : String
        Path to the file containing label data for training data
    architecture : String
        Name of the architecture for the model, valid inputs right now are "NASNet" and "Xception"
    trainData : String
        Path to folder containing training data


    Returns
    -------
    model : keras.model
        Tensorflow Keras model for Deep Learning applications. Trained with character data.
    le : sklearn.preprocessing.Label_Encoder
        Label Encoder used for initial encoding of the labels and for decoding in the output stage.

    """
    # Define learning parameters
    learnrate = 1e-1
    
    # Obtain both images and their corresponding labels
    # Read labels from a textfile
    labels = []
    tr_labelsf = open(labelfile, "r")
    tr_labels = tr_labelsf.readlines()
    tr_labelsf.close()
    for i in tr_labels:
        labels.append(i[:-1])
    labelnr = len(labels)
    
    # Read all images in image folder
    # !!!Label file has to correspond to the images in dataset folder for this to work correctly!!!
    data = []
    for i in range(0,len(labels)):
        filename = trainData + os.sep + str(i) + ".png" 
        image = cv2.imread(filename)
        image = cv2.resize(image, (16, 16))
        data.append(image)
    
    # Data Generation
    # Generate more data with added Salt and Pepper noise
    for i in range (0,40):
        for j in range (0,labelnr):
            newlabel = labels[j]
            noisyimage = data[j]
            out = np.copy(noisyimage)
            saltprob = 0.02 * i
            for k in range(out.shape[0]):
                for l in range(out.shape[1]):
                    rdn = np.random.rand()
                    if rdn < saltprob:
                        out[k][l] = [255,255,255]
                    elif rdn > 0.99:
                        out[k][l] = [0,0,0]
            noisyimage = out
            labels.append(newlabel)
            data.append(noisyimage)
            
    for i in range(0,len(data)):
        image = data[i]    
        image = cv2.resize(image, (width, height))
        data[i] = image
    # Convert data to model-suitable format
    data = np.array(data, dtype="float") / 255.0
    

    # Encode labels as integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels, numclasses)
    
    # Partition data into a training and a testing split
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    
    # Instantiate the Data Generator for training
    data_generator = ImageDataGenerator(width_shift_range=.5, height_shift_range=.1, zoom_range=[0.6,1.0])
    
    # Network generation
    if architecture == "NASNet":
        model = model_NasNet(width, height, epochs, learnrate, numclasses)
    elif architecture == "Xception":
        model = model_Xception(width, height, epochs, learnrate, numclasses)
    else:
        # Really bad, needs to be handled better
        assert 1 == 0, "No valid architecture provided"
    model.summary()
    
    log_dir = "eval/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Network Training
    H = model.fit(
    	x=data_generator.flow(trainX, trainY, batch_size=batchsize),
    	validation_data=(testX, testY),
    	steps_per_epoch=len(trainX) // batchsize,
    	epochs=epochs, callbacks=[tensorboard_callback])
    
    model.save("models/" + custom_name)
    with open("encoders/" + custom_name + ".sav", 'wb') as f:
        pickle.dump(le, f)

    return model, le