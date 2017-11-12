"""
    Basic SVM classifier.
    
    Classifier should support cross-validation and return stats including validation accuracy etc.

    Probably we can support train, test function later.
"""
import sklearn.svm
from sklearn.model_selection import cross_val_score

args = {'CUHKPQ': {'kernel':'rbf', 'C':10, 'class_weight':{1:2,0:1}}
        }

def svmBinaryCV(X,Y,**dummy):
    """
        Run cross-validation with svm on data.
        
        Args:
            X: N by M np array. N is the number of samples, M is length of features
            Y: N by 1 vector of label.
            dummy: additional args. not useful here.
        
        Return:
            validation accuracy.
    """
    clf = sklearn.svm.SVC(**args['CUHKPQ'])
    scores = cross_val_score(clf, X, Y, cv=5)
    return scores.mean()
