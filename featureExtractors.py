"""
    Base script for feature extractors
    
    feature script should take at least two inputs: np array of image and another arg (could be dummy if not used), and return 1d np array vector.
"""
from features import baseline, hueComposition, siftDesc, pedestrian, lines
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
              'pedestrian':
                    {'func':pedestrian.pedestrianDetector
                    },
              'lines':
                    {'func':lines.lines
                    }
            }
groupFeatureMap = {'bow':
                    {'pre': siftDesc.calcSIFTAll,
                    'trainfunc': siftDesc.trainKmeans,  # train func should take features, train a model dict with key 'name' and 'model'
                    'testfunc': siftDesc.assignGroup    # test func should take features and model (output from trainfunc), and give features
                    }
            }
featureToUse = ['baseline','hueComposition','pedestrian','lines']
groupFeatureToUse = 'bow'

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

def calcPreFeature(fileLists, featName = groupFeatureToUse):
    """
        calc feature required for group feature
    """
    if not featName:
        featName = groupFeatureToUse
    
    func = groupFeatureMap[featName]
    if 'pre' in func:
        return func['pre'](fileLists)
    else:
        return fileLists
        

def calcModel(features, featName = groupFeatureToUse):
    """
        calc model with features trained on images
    """
    if not featName:
        featName = groupFeatureToUse
    func = groupFeatureMap[featName]
    return func['trainfunc'](features)

def runModel(features, model, featName = groupFeatureToUse):
    """
        run the models on features to obtain (new) feature
    """
    if not featName:
        featName = groupFeatureToUse
    func = groupFeatureMap[featName]
    return func['testfunc'](features, model)