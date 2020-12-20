#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Florian Alkofer
"""
from typing import Tuple, Union
from deskew import determine_skew
from pytesseract import Output
import operator
import cv2
import numpy as np
import pytesseract
import math
import statistics

from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

# =============================================================================
"""
Preprocessing functions
An assortment of functions that are used to modify our images in order to remove types of noise or
add highlights for specific aspects.
"""

def cropScan(image):
    """
    Crops an image of a document scan as close to the page borders as possible to remove
    most border noise that occured during the scan proces. Double comented lines ar for bounding box checks.

    Parameters
    ----------
    image : image matrix
        Image data array.

    Returns
    -------
    retval : image matrix
        Cropped image data array.

    """
    #Preprocessing to reduce noise and highlight contours
    kernel = np.ones((3,3),np.uint8)
    contourIMG = cv2.fastNlMeansDenoising(image, 5, 7, 21)
    contourIMG = cv2.medianBlur(contourIMG, 5)
    contourIMG = cv2.morphologyEx(contourIMG.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    contourIMG = cv2.Canny(contourIMG, 30, 200)
    
    #Finding largest contour and coordinates for bounding rectangle
    contours, hierarchy = cv2.findContours(contourIMG, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    x,y,w,h = cv2.boundingRect(biggest_contour)
    
    ##Contour and bounding box checks, to see parameter adjustment results
    ##output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ##cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
    ##cv2.drawContours(output, biggest_contour, -1, 255, 3)
    ##cv2.imwrite('./shape_output.png', output)
    ##print("Height: {}".format(h))
    ##print("X: {}".format(x))
    ##print("Width: {}".format(w))
    ##print("Y: {}".format(y))
    
    retval = image[y:y+h, x:x+w]
    return retval


def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Rotates an image by previously determined angle and fills background with
    the provided color Information. Used to deskew text.
    Source: https://github.com/sbrunner/deskew

    Parameters
    ----------
    image : np.ndarray
        Image matrix to be deskewed.
    angle : float
        Angle the image is supposed to be rotated by.
    background : Union[int, Tuple[int, int, int]]
        Fill color for newly generated pixels.

    Returns
    -------
    image matrix
        Warped image according to deskew criteria.

    """
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def morphClose(image):
    """
    Morphological Closing Operator.

    Parameters
    ----------
    image : image matrix
        Base image.

    Returns
    -------
    image : image matrix
        Image with closing operator applied.

    """
    kernel = np.ones((3,3),np.uint8)
    image = cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return image


def morphOpen(image):
    """
    Morphological Opening Operator.

    Parameters
    ----------
    image : image matrix
        Base image.

    Returns
    -------
    image : image matrix
        Image with opening operator applied.

    """
    kernel = np.ones((3,3),np.uint8)
    image = cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return image


def globeThresh(image):
    """
    Apply global thresholding

    Parameters
    ----------
    image : image matrix
        Base image.

    Returns
    -------
    image : image matrix
        Globally thresholded base image.

    """
    ret, image = cv2.threshold(image, 127,255,cv2.THRESH_BINARY)
    return image


def otsu(image):
    """
    Apply Otsu thesholding

    Parameters
    ----------
    image : image matrix
        Base image.

    Returns
    -------
    image : image matrix
        Otsu thresholded base image.

    """
    image = image > threshold_otsu(image)
    return image


def niblack(image):
    """
    Apply Niblack thresholding

    Parameters
    ----------
    image : image matrix
        Base image.

    Returns
    -------
    binary_niblack : image matrix
        Niblack thresholded base image.

    """
    window_size = 25
    thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
    binary_niblack = image > thresh_niblack
    return binary_niblack


def sauvola(image):
    """
    Apply Sauvola thresholding

    Parameters
    ----------
    image : image matrix
        Base image.

    Returns
    -------
    image matrix
        Sauvola thresholded base image.

    """
    window_size = 35
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)
    image = image > thresh_sauvola
    return image.astype(np.uint8)


def canny(image):
    """
    Apply Canny Edge Detection, closing operator and pixel-inversion

    Parameters
    ----------
    image : image matrix
        Base image.

    Returns
    -------
    image : image matrix
        Base image with highlighted edges.

    """
    kernel = np.ones((3,3),np.uint8)
    image = cv2.Canny(image.astype(np.uint8),100,200)
    image = cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    image = cv2.bitwise_not(image)
    return image


def adapThresh(image):
    """
    Apply adaptive gaussian thresholding

    Parameters
    ----------
    image : image matrix
        Base image.

    Returns
    -------
    image : image matrix
        Adaptively thresholded base image.

    """
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return image


def compositeMethod(image):
    """
    Apply a composite of various preprocessing techniques to an image.
    These were selected for a specific use case for document image preprocessing
    and might not be fully suitable for all types of document images

    Parameters
    ----------
    image : image matrix
        Base image.

    Returns
    -------
    image : image matrix
        Processed image.

    """
    kernel = np.ones((3,3),np.uint8)
    image = cv2.medianBlur(image, 3)
    window_size = 25
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)
    image = image > thresh_sauvola
    #image = cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return image

# =============================================================================
"""
Helper Functions
Subdivisiion of workload sections for easier isolation of problem areas and for repeated usage.
"""

def angleDetection(image):
    """
    Determine angle of text in image.
    NOTE: Detection overshot or undershot by 90° for our use case. Thus we applied a small
        provisionary fix. If this fix leads to problems for other use cases comment out 
        the lines between "# Use case fix".

    Parameters
    ----------
    image : image matrix
        Text image.

    Returns
    -------
    angle : float
        Text skew angle in text image.

    """
    angle = determine_skew(image)
    
    # Use case fix
    while angle < -45:
        angle = angle + 90
    while angle > 45:
        angle = angle - 90
    # Use case fix
    
    return angle


def intersectBoxes(box1, box2):
    """
    Determine intersection area for bounding boxes.
    Original Source: https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    Parameters
    ----------
    box1 : List
        Bounding box parameters formatted like this: [x-coordinate, y-coordinate, width, height].
    box2 : List
        Bounding box parameters formatted like this: [x-coordinate, y-coordinate, width, height].

    Returns
    -------
    iou : float
        Percentage based intersection area.

    """
    #Box corner coordinates for axis aligned bounding boxes
    bb1x1 = box1[0]
    bb1x2 = box1[0]+box1[2]
    bb1y1 = box1[1]
    bb1y2 = box1[1]+box1[3]
    bb2x1 = box2[0]
    bb2x2 = box2[0]+box2[2]
    bb2y1 = box2[1]
    bb2y2 = box2[1]+box2[3]
    
    #Intersection rectangle coordinates
    x_left = max(bb1x1, bb2x1)
    y_top = max(bb1y1, bb2y1)
    x_right = min(bb1x2, bb2x2)
    y_bottom = min(bb1y2, bb2y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1x2 - bb1x1) * (bb1y2 - bb1y1)
    bb2_area = (bb2x2 - bb2x1) * (bb2y2 - bb2y1)

    if bb1_area == 0 or bb2_area == 0:
        return 1.0

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
    

def horizontalProjection(box, image):
    """
    Create a projection along the x-axis of a box

    Parameters
    ----------
    box : List
        Bounding box parameters formatted like this: [x-coordinate, y-coordinate, width, height].
    image : image matrix
        Image corresponding to the bounding box coordinates.

    Returns
    -------
    projection : List
        Count of how many nonwhite pixels are in each column.

    """
    projection = []
    for i in range(0,box[2]):
        y_list = []
        for j in range(0, box[3]):
            y_list.append(255 - image[box[1] + j, box[0] + i])
        projection.append(sum(y_list))
    return projection


def boxMerge(box1, box2):
    """
    Create a bounding box that contains two given boxes

    Parameters
    ----------
    box1 : List
        Bounding box parameters formatted like this: [x-coordinate, y-coordinate, width, height].
    box2 : List
        Bounding box parameters formatted like this: [x-coordinate, y-coordinate, width, height].

    Returns
    -------
    List
        Bounding box parameters formatted like this: [x-coordinate, y-coordinate, width, height].

    """
    return (min(box1[0], box2[0]), 
         min(box1[1], box2[1]), 
         max(box1[2] + box1[0], box2[2] + box2[0]),
         max(box1[3] + box1[1], box2[3] + box2[1]))


def mergeOverlap(boxList):
    """
    Detect boxes within a list that overlap with each other and generate a new
    box that contains both overlapping boxes.

    Parameters
    ----------
    boxList : List
        List containing multiple bounding boxes.

    Returns
    -------
    boxList : List
        List containing multiple bounding boxes.

    """
    overlapThreshold = 0.0
    for i in range(0, len(boxList)):
        for j in range(i, len(boxList)):
            intersections = []
            if i != j:
                if intersectBoxes(boxList[i],boxList[j]) > overlapThreshold:
                    intersections.append(j)
        if len(intersections) > 0:
            mergeBox = boxList[i]
            newBoxList = boxList
            for k in reversed(intersections):
                mergeBox = boxMerge(mergeBox, boxList[k])
                newBoxList.pop(k)
            newBoxList.pop(i)
            return mergeOverlap(newBoxList)
    return boxList


def filterBoxes(results):
    """
    Use criteria like height or content to filter bounding boxes that are not
    suited for our analysis out.

    Parameters
    ----------
    results : List
        Results from pytesseract.image_to_data().

    Returns
    -------
    outputList : List
        Reduced input results.

    """
    # compute median height for all viable bounding boxes for the page
    heightList = []
    for i in range(0, len(results["text"])):
 	     if int(results["conf"][i])>-1:
              heightList.append(results["height"][i])
    height = statistics.median(heightList)    
    # filter boxes too far apart from our median height and convert boxes to [x,y,w,h] format
    outputList = []
    for i in range(0, len(results["text"])):
        if (
                0.7*height <= results["height"][i] <= 1.4*height and
                int(results["conf"][i])>-1
                ):
            outputList.append((results["left"][i], results["top"][i], 
                                results["width"][i], results["height"][i]))
    return outputList
    

def addBoxes(boxLBase, boxLMerge):
    """
    Add a group of bounding boxes to another without causing intersections
    within the result group.

    Parameters
    ----------
    boxLBase : List
        Base list of bounding boxes.
    boxLMerge : List
        List of bounding boxes to be added to the base list.

    Returns
    -------
    boxLBase : List
        New list of bounding boxes.

    """
    for i in range (0, len(boxLMerge)):
        for j in range (0, len(boxLBase)):
            if intersectBoxes(boxLMerge[i], boxLBase[j]):
                break
        # If for-loop finished without break we can append the list since no intersection was detected
        # but if the break instruction is executed we will skip straight to the next index in the outer loop
        else:
            boxLBase.append(boxLMerge[i])
    return boxLBase



def characterSegmentation(wordBoxes, image):
    """
    Segment a group of candidate word bounding boxes into character bounding boxes.

    Parameters
    ----------
    wordBoxes : List
        List of word-level bounding boxes.
    image : image matrix
        Image corresponding to the bounding boxes.

    Returns
    -------
    characterBoxes : List
        List of character-level bounding boxes.

    """
    
    # Create horizontal projection table which highlights along which x-values inside a 
    # given box no or only few non white pixels are to find letter boundaries 
    projectionTable = []
    for i in range(0, len(wordBoxes)):
        projectionTable.append(horizontalProjection(wordBoxes[i], image))
    
    # Count lengths of blank and non blank pixels in a row to find out median length of
    # letter and blank spaces between letters for separation
    characterLengths = []
    spaceLengths = []
    for i in range (0, len(projectionTable)):
        characterPixels = 0
        blankPixels = 0
        projection = projectionTable[i]
        for j in range (0, len(projection)):
            if projection[j] > 0:
                characterPixels += 1
                if blankPixels > 0:
                    spaceLengths.append(blankPixels)
                    blankPixels = 0
            else:
                blankPixels += 1
                if characterPixels > 0:
                    characterLengths.append(characterPixels)
                    characterPixels = 0
        if characterPixels > 0:
            characterLengths.append(characterPixels)
            characterPixels = 0
        if blankPixels > 0:
            spaceLengths.append(blankPixels)
            blankPixels = 0
    if not characterLengths:
        characterLengths = 0
    else:
        characterMedian = statistics.median(characterLengths)
    if not spaceLengths:
        blankMedian = 0
    else:
        blankMedian = statistics.median(spaceLengths)

    characterDist = characterMedian + blankMedian
    #print(characterDist)
    minCharacterDist = (characterDist / 2) + 2
    minCharacterDist = int(minCharacterDist)
    #print(minCharacterDist)
    maxCharacterDist = max(minCharacterDist,characterDist) + 2
    maxCharacterDist = int(maxCharacterDist)
    #print(maxCharacterDist)

    # Character Segmentation
    # We split each bounding box into at least 'minCharacterDist' long pieces 
    # along where our projection found blank spaces
    characterBoxes = []

    for i in range(0,len(wordBoxes)):
        workingBox = wordBoxes[i]
        workingProjection = projectionTable[i]
        index = 0
        beginIndex = 0
        while index < workingBox[2] - minCharacterDist:
            index += minCharacterDist
            while index < len(workingProjection):
                curBoxLength = index - beginIndex
                if workingProjection[index] == 0 or curBoxLength > maxCharacterDist:
                    characterBoxes.append((workingBox[0] + beginIndex, workingBox[1],
                                   index - beginIndex, workingBox[3]))
                    beginIndex = index
                    break
                else:
                    index += 1
        else:
            characterBoxes.append((workingBox[0] + beginIndex, workingBox[1],
                                   workingBox[2] - beginIndex, workingBox[3]))
    
    return characterBoxes


def boxOrder(characterList):
    """
    Create a structure that sorts each character-level into line-level and
    further into word-level categories for easier assignment.

    Parameters
    ----------
    characterList : List
        List of character-level bounding boxes.

    Returns
    -------
    lineList : List
        List of character-level bounding boxes further divided into word- and line-
        level lists.

    """
    assert len(characterList)>1
    
    #Sort character boxes into their corresponding lines
    lineList = []
    ycoordList = []
    ycoordList.append(characterList[0][1])
    lineList.append([])
    lineList[0].append(characterList[0])
    
    heightList = []
    for i in range(0, len(characterList)):
 	     heightList.append(characterList[i][3])
    avgheight = statistics.median(heightList)    
    
    for i in range(1,len(characterList)):
        ymin = characterList[i][1] - (avgheight*(2/3))
        ymax = characterList[i][1] + (avgheight*(2/3))
        for j in range (0,len(ycoordList)):
            if ymin < ycoordList[j] and ymax > ycoordList[j]:
                lineList[j].append(characterList[i])
                break
        else:
            ycoordList.append(characterList[i][1])
            lineList.append([])
            lineList[len(ycoordList)-1].append(characterList[i])
    
    #Sort each line further into words
    for i in range(0,len(lineList)):
        workingLine = lineList[i]
        wordOrderLine = []
        
        while len(workingLine)>0:
            #Find furthest left bounding box
            xMin = 1000000
            boxIndex = 0
            for j in range(0,len(workingLine)):
                curBox = workingLine[j]
                if curBox[0] < xMin:
                    xMin = curBox[0]
                    boxIndex = j
            
            word = [workingLine[boxIndex]]
            workingLine.remove(workingLine[boxIndex])
            incomplete = 1
            lastBox = word[0]
            
            #Continue to add bounding boxes to the current word until we no longe find any adjacent ones
            while incomplete:
                for j in range(0,len(workingLine)):
                    curBox = workingLine[j]
                    lastBoxEnd = lastBox[0] + lastBox[2]
                    if curBox[0] == lastBoxEnd:
                        word.append(curBox)
                        lastBox = curBox
                        workingLine.remove(workingLine[j])
                        break
                else:
                    incomplete = 0            
            wordOrderLine.append(word)
        
        lineList[i] = wordOrderLine
    return lineList



# =============================================================================

def docImageSegment(inputImage, crop, deskew, segIllust):
    """
    Preprocesses and segments a document image on character level by exploiting
    structural features.

    Parameters
    ----------
    inputImage : numpy.array
        Image file converted to numpy.array to be compatible with OpenCV operations.
    crop : int
        Determines if input image is supposed to be cropped. Only use if discernible
         page border was present in base image file
    skew : int
        Determines if input image is supposed to be corrected for text skew. Errorprone
        so only use if confident.


    Returns
    -------
    characterBoxes : List
        List of bounding boxes for each character in [x,y,width,height]-format.
        Sorted into smaller sublists for each line and word.
    imageList : List
        List of processed images.

    """
    
    baseImage = inputImage
    basaImage = baseImage
    if deskew:
        textAngle = angleDetection(baseImage)
        deskewedImage = rotate(baseImage, textAngle, (255,255,255))
        basaImage = deskewedImage
    if crop:
        basaImage = cropScan(deskewedImage)
    
    # Second preprocessing stage
    blurredImage = cv2.medianBlur(basaImage, 3)
    openPostImage = compositeMethod(basaImage)
    openPreImage = cv2.morphologyEx(blurredImage.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    openPreImage = sauvola(openPreImage)
    erodePostImage = sauvola(blurredImage)    
    erodePostImage = cv2.erode(erodePostImage, np.ones((3,3),np.uint8))

    resultsBase = pytesseract.image_to_data(openPostImage, output_type=Output.DICT)
    resultsOpen = pytesseract.image_to_data(openPreImage, output_type=Output.DICT)
    resultsErode = pytesseract.image_to_data(erodePostImage, output_type=Output.DICT)
    
    # Filter box selection by preset criteria (height, confidence)
    boxListBase = filterBoxes(resultsBase)
    boxListOpen = filterBoxes(resultsOpen)
    boxListErode = filterBoxes(resultsErode)
    
    # Afterwards merge boxes onto the base list (simple approach)
    boxListBase = addBoxes(boxListBase, boxListOpen)
    boxListBase = addBoxes(boxListBase, boxListErode)

    cpyImg = basaImage
    cpyImg = cv2.cvtColor(cpyImg,cv2.COLOR_GRAY2RGB)
    for box in boxListBase:
        x, y, w, h = box
        cv2.rectangle(cpyImg,  (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite("eval/word_"+ str(segIllust) + ".png", cpyImg)
        
    # Sort boxes by y-coordinate(primary key, topmost first) and x-coordinate(secondary key, leftmost first)
    boxListBase.sort(key = operator.itemgetter(1, 2))

    # Separate larger word boxes into smaller character based boxes
    characterBoxes = characterSegmentation(boxListBase, basaImage)
    orderedCharacters = boxOrder(characterBoxes)
    # Required to un-normalize the preprocessed images
    imageList = [basaImage]
    
    for i in range(0,len(imageList)):
        imageList[i] = cv2.cvtColor(imageList[i],cv2.COLOR_GRAY2RGB)
    
    # Create image for evaluation of segmentation
    lstcpyImg = basaImage
    lstcpyImg = cv2.cvtColor(lstcpyImg,cv2.COLOR_GRAY2RGB)
    for i in range(0, len(orderedCharacters)):
        line = orderedCharacters[i]
        for j in range(0, len(line)):
            word = line[j]
            for k in range(0, len(word)):
                x, y, w, h = word[k]
                cv2.rectangle(lstcpyImg,  (x, y), (x + w, y + h), (0, 255, 0), 2)        
    cv2.imwrite("eval/char_"+ str(segIllust) + ".png", lstcpyImg)
    
    return orderedCharacters, imageList