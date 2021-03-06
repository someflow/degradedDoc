# Transcription of typewritten documents

This package tries to improve Tesseract's OCR results for typewritten degraded document images by utilizing neural network predictions on a character level.

It is concepted to be as specialized for this case as possible for more accurate transcriptions of those documents.

## Pretrained model repository

A pretrained model, its encoder and our training data are available at: (https://drive.google.com/drive/folders/1LdJLTLEWIFK6Zc8lHAE9Ku2MXzqM2GC1?usp=sharing)

## Installation

Not uploaded to any index yet

### Packages

The packages required to use this package are:

* **deskew**
* **pytesseract**
* **cv2**
* **numpy**
* **scikit-learn**
* **tensorflow**
* **scikit-image**
* **pdf2image**

### Other Software requirements

To use this package properly, you are also required to have working installations of the following softwares:

#### Tesseract

Installation guide: (https://tesseract-ocr.github.io/tessdoc/Installation.html)

The base OCR engine used for text detection in this project.

#### Poppler

An installation guide can be found here, briefly at the top: (https://pypi.org/project/pdf2image/) 

Used for the pdf2image package, to enable PDF-input.

#### Fontforge-Python

Windows: (https://stackoverflow.com/questions/23365299/how-to-import-fontforge-to-python-in-windows-7)

Ubuntu:

```bash
sudo apt-get install fontforge python-fontforge
```

Required to use the training data generation script in case you want to train your own networks.

## Usage example

```python
#if you installed this as a package you will need to #import degradedDoc.degradedDoc
import degradedDoc

# Transcribe a file called  "sample.pdf" using the default set of arguments
transcript1 = degradedDoc.degradedDocOCR(inputfile = "sample.pdf")

# Transcribe an image suing a custom model
transcript2 = degradedDoc.degradedDocOCR(inputfile = "image.png", modelpath = "models/Custom", labelencpath = "encoders/custom.sav")

# Retrain a custom model during runtime with already set-up training data, labelfile is in
# executing directory and called "labels.txt" while corresponding training data is located in directory "dataset"
transcript3 = degradedDoc.degradedDocOCR(inputfile = "image.png", retrain = 1, architecture = "Xception", 
										labelfile = "labels.txt", trainData = "dataset", numclasses = 77)

# Train a model with previous parameters and store both model and encoder under "models/newModel" and "encoder/newModel" respectively
degradedDoc.trainModel(custom_name = "newModel", width = 150, height = 150, architecture = "Xception", 
										labelfile = "labels.txt", trainData = "dataset", numclasses = 77)
```
## Training Data

You are required to prepare a set of training data to train a network with out script in a specific format, which is:

* Character Images in PNG format numbered 1 to n, e.g. "1.png", "2.png", ..., "222.png"
* Text file containing the true label for the character images whereas each line corresponds to the file name, e.g. if "1.png" is an image of "A" the first line of "labels.txt" should be just "A"

### Prepare training data

There is also a small script which fully prepares the training data for usage with our system. It requires fontforge to be used.

Execute the trainingDataGen.py script using Fontforge's built in Python environment. For this also you are required to have a folder called "fonts" in your working directory containing various TrueType-Font files from which you want to extract characters. In Ubuntu this can be done with:

```bash
ffpython trainingDataGen.py
```

## Outputs

Evaluation outputs visualizing the segmentations and detected text in the document image(s) are put out to an evaluation folder. The main method of the package returns a String containg the full predicted textual transcription.

