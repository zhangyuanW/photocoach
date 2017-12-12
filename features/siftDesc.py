"""
    Produce SIFT descriptor 
"""
import cv2
from hueComposition import resize
import numpy as np
from sklearn.cluster import KMeans
from scipy.misc import imread
K = 200

def calcSIFT(img):
    """
        calculate sift descriptors in image
    """
    if len(img.shape)==3:
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = resize(gray,min(1024,max(gray.shape[0],gray.shape[1])))
    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des = sift.detectAndCompute(gray,None)
    return des
    
def calcSIFTAll(fileLists):
    """
        calc sift for multiple images
    """
    res = []
    print ('start calc sift')
    for i,f in enumerate(fileLists):
        res += [calcSIFT(imread(f))]
        if res[-1] is None:
            print ('no keypoints in '+f)
        if i%100 == 0:
            print("{0} out of {1} sift done".format(i,len(fileLists)))
    return res

def trainKmeans(SIFTs):
    """
        Give sift features, train a KMeans model
    """
    features = np.concatenate([s[:500] for s in SIFTs if s is not None], axis=0)
    features = features[::10]
    print ("running kmeans on sift samples {0}".format(len(features)))
    kmeans = KMeans(n_clusters=K, random_state=0).fit(features)
    return kmeans
    
def assignGroup(SIFTs, kmeans):
    """
        Given features and kmeans model, assign feat to c, calc hist as features
    """
    res = []
    for sifts in SIFTs:
        if sifts is None:
            groups = [0]*len(kmeans.cluster_centers_)
        else:
            groups = kmeans.predict(sifts)
            # groups = [min((sum((ss-cc)**2 for ss,cc in zip(sft,c)),ic)\
                        # for ic,c in enumerate(centers))[0]\
                        # for sft in sifts]
            thisres = [0]*len(kmeans.cluster_centers_)
            for g in groups:
                thisres[g] += 1.0/len(sifts)
        res += [np.array(thisres)]
    return res
    