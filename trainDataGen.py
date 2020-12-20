#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The only way to execute this script as of now is to use the Fontforge's python environment
This means you will have to install Fontforge and then manually open its python environment
and from there navigate to this script and execute it with:
    ffpython trainDataGen.py

Guide for Windows: https://stackoverflow.com/questions/23365299/how-to-import-fontforge-to-python-in-windows-7

This script creates training data and a label file for training with the degradedDocOCR-package.

To add more fonts to be extracted from add the full filepath to the "fontset" list
To add more characters to be extracted add their .ttf-name to the "charSet" list in "isInCharacterSet"
@author: Florian Alkofer
"""
import fontforge
import os

def createTrainData(fontdir = "fonts", outputdir = "trainData"):
    """
    Generate training data for model training by extracting character images from ttf files

    Parameters
    ----------
    fontdir : string
        Directory containing .ttf files

    outputdir : string
        Directory in which trainingdata will be stored.

    Returns
    -------
    None.
    """
    charSet = ["a", "A", "b", "B", "c", "C", "d", "D", "e", "E", "f", "F", "g", "G", "h", "H", "i", "I", 
               "j", "J", "k", "K", "l", "L", "m", "M", "n", "N", "o", "O", "p", "P", "q", "Q", "r", "R",
               "s", "S", "t", "T", "u", "U", "v", "V", "w", "W", "x", "X", "y", "Y", "z", "Z", 
               "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", 
               "adieresis", "Adieresis", "odieresis", "Odieresis", "udieresis", "Udieresis",
               "period", "question", "exclam", "colon", "minus", "parenleft", "parenright", "plus", "space"]

    specialChars = {"zero" : "0", "one" : "1", "two" : "2", "three" : "3", "four" : "4", "five" : "5", "six" : "6", "seven" : "7", "eight" : "8", "nine" : "9", 
                    "adieresis" : "ä", "Adieresis" : "Ä", "odieresis" : "ö", "Odieresis" : "Ö", "udieresis" : "ü", "Udieresis" : "Ü",
                    "period" : ".", "question" : "?", "exclam" : "!", "colon" : ":", "minus" : "-", "parenleft" : "(", "parenright" : ")", "plus" : "+", "space" : " "}


    fontset = []

    for file in os.listdir(fontdir):
        if file.endswith(".ttf") or file.endswith(".TTF"):
            fontset.append(file)

    counter = 0
    labelfile = open(outputdir + os.sep + "labels.txt" ,"w+")
    print(fontset)
    for font in fontset:    
        F = fontforge.open(fontdir + os.sep + font)
        for name in F:
            if name in charSet:
                filename = outputdir + os.sep + str(counter) + ".png"
                counter = counter + 1
                # print name
                F[name].export(filename)

                if name in specialChars:
                    labelfile.write(specialChars[name] + "\n")
                else:
                    labelfile.write(name + "\n")
       
    labelfile.close()

createTrainData()