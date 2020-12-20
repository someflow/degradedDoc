#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Florian Alkofer
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

import numpy as np
import argparse
import cv2
import os


def output_gen(width, height, model, labelEnc, boundingBoxes, docImgList):
    """
    Generates predictions for a set of bounding boxes for corresponding images
    based on a pre-trained model and produces an output through majority voting.

    Parameters
    ----------
    width : int
        Width that images are resized to for prediction. Should match width model
        was trained with
    height : int
        Height that images are resized to for prediction should match height model
        was trained with
    model : keras.model
        A Neural Network from the keras module that has been trained and compiled already.
    labelEnc : sklearn.preprocessing.LabelEncoder
        Corresponding Label Encoder for the provided model.
    boundingBoxes : List
        Three dimensional list containing bounding boxes for characters extracted from a document image
        in order.
    docImgList : List
        List of images with various forms of preprocessing applied to them.

    Returns
    -------
    String
        String of the transcribed text

    """
    output = ""
    
    for line in range(0,len(boundingBoxes)):
        curLine = boundingBoxes[line]
        for word in range(0,len(curLine)):
            curWord = curLine[word]
            for character in range(0,len(curWord)):
                curChar = curWord[character]
                
                #Obtain Prediction for the current character
                images = []
                for img in docImgList:
                    imag = img[curChar[1]:curChar[1]+curChar[3], curChar[0]:curChar[0]+curChar[2]]
                    imag = cv2.resize(imag, (width, height))
                    images.append(imag)
                images = np.array(images, dtype="float") / 255.0
                #cv2.imwrite("eval/pred" + str(line) + " " + str(word) + " " + str(character) + ".png", images[0]*255.0)
                prediction = model.predict(images)
                prediction = np.argmax(prediction, axis=1)
                counts = np.bincount(prediction)
                highest = np.argmax(counts)
                highestar = [highest]
                labelInverse = labelEnc.inverse_transform(highestar)
                output = output + str(labelInverse[0])
            
            if word < len(curLine)-1:
                output = output + " "
        if line < len(boundingBoxes)-1:
            output = output + "\n"
            
    return output