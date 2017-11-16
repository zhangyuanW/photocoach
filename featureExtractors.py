"""
    Base script for feature extractors
    
    feature script should take at least two inputs: np array of image and another arg (could be dummy if not used), and return 1d np array vector.
"""
from features import baseline, hueComposition, siftDesc, lines
import numpy as np

# add more features here
featureMap = {'baseline': 
                    {'func':baseline.resizeAndVec,  # use the key `func`
                    'kwargs': {'img_size':32}         # use the key `kwargs` and pass in keys as in your feature function. Besides this kwargs, function in this script should take at least one input (image). 
                    },
              'hueComposition':
                    {'func':hueComposition.hueCompose
                    },
              'sift':
                    {'func':siftDesc.calcSIFT
                    },
              'lines':
                    {'func':lines.lines
                    }
            }
featureToUse = ['baseline','hueComposition', 'lines']

def calcFeatures(image, feats = featureToUse):
    """
        Run the features in featureMap, with each func and args
        
        Args:
            image: np array data of image
            feats: feat names to use. By default, use what is in featureToUse
        
        Return:
            feature vector concatenated from all features
    """
    if not feats:
        feats = featureToUse
    res = []
    for featName in feats:
        funcAndArgs = featureMap[featName]
        if 'kwargs' in funcAndArgs:
            res.append(funcAndArgs['func'](image,**funcAndArgs['kwargs']))
        else:
            res.append(funcAndArgs['func'](image))
    return np.concatenate(res)
