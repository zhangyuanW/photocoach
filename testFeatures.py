"""
    This will take a featureName and do basic testing, visualization.
"""
from util_IO import getFeatLabel
import matplotlib.pyplot as plt

def testHist(data_dir, featureName, num=None):
    """
        Read images, calc feature and plot on histogram. The feature MUST be 1d.
        
        Args:
            data_dir: path to dataset
            num: if specified, read num of positive and negative images
            featureName: name of one feature to be tested.
            
        Return:
            X: feature matrix
            Y: label
            imNames: names of images.
            
        Usage:
            import testFeatures
            X,Y,names = testFeatures.testHist(data_dir,'hueComposition',10)
    """
    X,Y,imNames = getFeatLabel(data_dir,num,[featureName])
    
    plt.hist(X[Y==1],bins=100,alpha=0.5,label = 'high')
    plt.hist(X[Y==0],bins=100,alpha=0.5, label = 'low')
    plt.legend()
    plt.show()
    
    return X,Y,imNames
