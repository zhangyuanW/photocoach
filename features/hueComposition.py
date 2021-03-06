import numpy as np
import collections
import cv2
import colorsys
import operator
import imutils
from matplotlib import pyplot as plt

def resize(image, length):
    height = image.shape[0];
    width = image.shape[1];
    
    if height > width:
        width = int(width*length/height)
        height = length
    else:
        height = int(height*length/width)
        width = length
    
    return cv2.resize(image, (width, height));

def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

def hueCompose(image):
    #img = cv2.imread('639328.jpg')
    image = resize(image, 200)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        
    Z = image.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    counter = collections.defaultdict(int);
    for la in label:
        counter[la[0]] += 1;

    sorted_x = sorted(counter.items(), key=operator.itemgetter(1))
    color1 = center[sorted_x[K-1][0]];
    color2 = center[sorted_x[K-2][0]];

    hsv1 = rgb2hsv(color1[0],color1[1],color1[2]);
    hsv2 = rgb2hsv(color2[0],color2[1],color2[2]);

    result = abs(hsv2[0]-hsv1[0]);
    if result > 180:
        result = 360-result;
    
    return np.array([result])
    
    if goodCompose(result):
        return np.array([result,1,0]);
    return np.array([result,0,1]);

def goodCompose(result):
    if result <= 10 or result >= 170:
        return True;
    else:
        return False;