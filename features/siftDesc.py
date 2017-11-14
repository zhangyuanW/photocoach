"""
    Produce SIFT descriptor 
"""
import cv2

def calcSIFT(img):
    """
        calculate sift descriptors in image
    """
    if len(img.shape)==3:
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    return des
