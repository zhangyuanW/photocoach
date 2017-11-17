"""
    This will take a featureName and do basic testing, visualization.
"""
from util_IO import getFeatLabel
import matplotlib.pyplot as plt
import numpy as np

def testHist(data_dir, featureName, num=None, normalize = True):
    """
        Read images, calc feature and plot on histogram. The feature MUST be 1d.
        
        Args:
            data_dir: path to dataset
            featureName: name of one feature to be tested.
            num: if specified, read num of positive and negative images
            normalize: default true. normalize the bar height of each class to the number of samples in that class.
            
        Return:
            X: feature matrix
            Y: label
            imNames: names of images.
            
        Usage:
            import testFeatures
            X,Y,names = testFeatures.testHist(data_dir,'hueComposition',10)
    """
    # read images and labels
    images, labels, names = readImageLabel(data_dir)
    
    # calc individual features
    indFeats = getIndividualFeat(images,[featureName])
    
    #X,Y,imNames = getFeatLabel(data_dir,num,[featureName])
    if normalize:
        myarray = X[Y==0]
        weights = np.ones_like(myarray)/float(len(myarray))
        plt.hist(myarray, weights=weights,alpha=0.5, label = 'low')
        myarray = X[Y==1]
        weights = np.ones_like(myarray)/float(len(myarray))
        plt.hist(myarray, weights=weights,alpha=0.5,label = 'high')
    else:
        plt.hist(X[Y==0], alpha=0.5, label = 'low')
        plt.hist(X[Y==1], alpha=0.5, label = 'high')
        
    plt.legend()
    plt.show()
    
    return X,Y,imNames
