from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2

def pedestrianDetector(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    image = imutils.resize(image, width=min(400, image.shape[1]))

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    nums = len(pick);
   
    total = 0;
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        total += abs(xA-xB)*abs(yA-yB);
        
    imageSize = image.shape[0]*image.shape[1]
    
    result = [nums, total/imageSize]
    return np.array(result)
