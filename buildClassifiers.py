"""
    Base script for training and testing classifiers
    
    Classifier should take at least three inputs, data matrix X and label Y, and another arg (could be dummy if not used). The functin need to support cross-validation and return stats such as validation accuracy.

    Probably we can support train, test function later.
"""

from classifiers import svm
import numpy as np

# add more classifiers here
classifierMap = {'svm': 
                    {'func':svm.svmBinaryCV,  # use the key `func`
                    'kwargs': {}         # use the key `args`. Function in this script should take at least two input (X and Y)
                    }
            }
            
classifierToUse = ['svm']

def runBinaryCV(X,Y):
    """
        Run cross-validation on binary classifier on data X and label Y
        
        Args:
            X: N by M np array. N is the number of samples, M is length of features
            Y: N by 1 vector of label (0,1).
            **funcAndArgs['kwargs']: dummy variable to stay consistent with input...
        Return:
            validation accuracy.
    """
    res = []
    for className in classifierToUse:
        funcAndArgs = classifierMap[className]
        thisres = funcAndArgs['func'](X, Y, **funcAndArgs['kwargs'])
        res.append(thisres)
        print('{0} return accuracy {1}'.format(className, thisres))
    return np.array(res)
    