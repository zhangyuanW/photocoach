"""
    Baseline feature extractor. 
   
"""
import numpy as np
from scipy.misc import imresize,imread

def rgb2gray(rgb):
    """
        Helper function to convert color image to grayscale
    """
    if len(rgb.shape) == 3:
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    else:
        return rgb
    
def resizeAndVec(image,img_size=32):
    """
        Take an image, resize it to img_size*img_size (grayscale), and return a vector.
        
        Args:
            image: np array, supposed to be w*h*3 or w*h
            img_size: int. image will be resize to img_size by img_size
            
        Return:
            vectorized image of length img_size*img_size. Type np.array. Shape (dim,)
    """
    return imresize(rgb2gray(image),(img_size,img_size)).reshape(img_size*img_size)/255.0 - 0.5
