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
    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des = sift.detectAndCompute(gray,None)
    return des
