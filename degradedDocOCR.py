"""
TUM Master's Thesis: Reconstruction of Degraded Historical 
Text Documents using Deep Learning

Usage of this script is for automatic segmentation
of a scan image of a degraded physical text document
and further prediction for the value of each extracted
character using a Neural Network.
"""

__author__ = "Florian Alkofer"
__copyright__ = "Copyright 2020"
__license__ = "All rights reserved"
__maintainer__ = "Florian Alkofer"
__email__ = "flo.alkofer@tum.de"
__status__ = "Work in Progress"


import modeltraining
import masterProcessing
import outputGen
import os
import cv2
import pickle
import numpy as np
#import trainDataGen
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)



def degradedDocOCR(inputfile, height = 150, width = 150, labelfile = 'trainData/labels.txt', trainData = "trainData",
                     batch_size = 32, epochs = 50, crop = 0, deskew = 1, retrain = 0, architecture = "Xception", numclasses = 77,
                     modelpath = "models/Xception", labelencpath = "encoders/Xception.sav", learnrate = 1e-1):
    """
    Main script for execution of the reconstruction process.
    
    Parameters
    ----------
    inputfile : String
        Path to the file that will be transcribed
    height : int
        Height for the input layer of the network
    width : int
        Width for the input layer of the network
    labelfile : String
        Path to the file containing labels for the training data
    trainData : string
        Path to directory containing training data
    batch_size : int
        Batch size for network training
    epochs : int
        Number of epochs the network will be trained for
    crop : int
        Used if the document image should be reduced to a page frame. Only use if there is a clearly visible page border in the document image.
    deskew : int
        Used to correct angled text in document images. Disable if pages are rotated the wrong direction
    retrain : int
        Enables retraining of the network at runtime. Takes a long time and requires properly setup training data.
    architecture : String
        Determines the architecture used for predictions and training. Currently only "Xception" is supported
    numclasses : int
        Number of different classes in training data set
    modelpath : String
        Path to the directory containing the saved model for loading pretrained models
    labelencpath : String
        Path to the labelencoder corresponding to a pretrained network for en- and decoding of character labels
    learnrate : float
        Learnrate for training of the network

    Returns
    -------
    String
        Transcribed content from input document pages

    """
   
    images = []
    if inputfile.endswith(".pdf"):
        pilImg = convert_from_path(inputfile)
        for img in pilImg:
            img = img.convert("L")
            openCvimg = np.array(img)
            openCvimg = (255 - openCvimg)
            _,openCvimg = cv2.threshold(openCvimg,30,255,cv2.THRESH_TOZERO)
            openCvimg = (255 - openCvimg)
            images.append(openCvimg)
    elif inputfile.endswith(".png"):
        img = cv2.imread(inputfile, 0)
        images.append(img)
    elif inputfile.endswith(".jpg"):
        img = cv2.imread(inputfile, 0)
        images.append(img)
    else:
        print("Incompatible input file type. Only use PDF, JPG or PNG files for OCR.")
        return ""

    # Segment document image(s)
    bounds = []
    docImages = []
    imgCount = 0
    for img in images:
        boxes, docimgs = masterProcessing.docImageSegment(img, crop, deskew, imgCount)
        bounds.append(boxes)
        docImages.append(docimgs)
        imgCount = imgCount + 1
    
    # Obtain model and label Encoder
    if retrain:
        model, labelEncode = modeltraining.train_model(width = width, height = height, architecture = architecture,
                                                         epochs = epochs, learnrate = learnrate, numclasses = numclasses, batchsize = batch_size, trainData = trainData, labelfile = labelfile)
    else:
        model = keras.models.load_model(modelpath)
        with open(labelencpath, 'rb') as f:
            labelEncode = pickle.load(f)
    
    # Generate transcription
    output = ""
    for i in range(0,len(bounds)):
        output = output + outputGen.output_gen(width, height, model, labelEncode, bounds[i], docImages[i])
        output = output + "\n"

    return output


def trainModel(custom_name, width, height, epochs = 50, learnrate = 1e-1, numclasses = 77, labelfile = "trainData/labels.txt", architecture = "Xception", trainData = "trainData", batchsize = 32):
    """
    Trains a network for character image predictions and stores a label encoder and the network.
    
    Parameters
    ----------
    height : int
        Height for the input layer of the network
    width : int
        Width for the input layer of the network
    epochs : int
        Number of epochs the network will be trained for
    learnrate : float
        Learnrate for training of the network
    numclasses : int
        Number of different classes in training data set
    labelfile : String
        Path to the file containing labels for the training data
    architecture : String
        Determines the architecture used for predictions and training. Currently only "Xception" is supported
    trainData : string
        Path to directory containing training data
    batch_size : int
        Batch size for network training

    Returns
    -------
    None.

    """
    # Find a way to prepare Training data without using Fontforge
    #trainDataGen.createTrainData()
    modeltraining.train_model(width, height, epochs, learnrate, numclasses, labelfile, architecture, trainData, batchsize, custom_name)