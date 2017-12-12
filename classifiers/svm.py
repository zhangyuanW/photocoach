"""
    Basic SVM classifier.
    
    Classifier should support cross-validation and return stats including validation accuracy etc.

    Probably we can support train, test function later.
"""
import sklearn.svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import chi2_kernel, additive_chi2_kernel, paired_manhattan_distances
import numpy as np
import copy
from IPython.core.debugger import Tracer
import math

args = {'CUHKPQ': {'kernel':'rbf', 'degree':1, 'C':10, 'class_weight':'balanced'},
        'SVM': {'kernel':chi2_kernel, 'degree':1, 'C':100, 'class_weight':'balanced'}
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
    
def svmTrain(X,Y,param = None,regression=False):
    """
        train svm model and return model
    """
    if param == None:
        param = args['CUHKPQ']
    if regression:
        param = copy.deepcopy(param)
        param.pop('class_weight',1)
        clf = sklearn.svm.SVR(**param)
    else:
        clf = sklearn.svm.SVC(**param)
    clf.fit(X, Y)
    score = clf.score(X,Y)
    predict = clf.predict(X)
    print(score, math.sqrt(((predict-Y)**2).mean()))
    # Tracer()()
    
    return clf

def svmTest(clf,X,Y, featNames, regression = False):
    """
        test svm model, return accuracy
    """
    print featNames
    if not regression:
        predict = clf.predict(X)
        print confusion_matrix(Y,predict)
        accu = np.mean(predict==Y)            
        print accu
        return accu,predict
    else:
        score = clf.score(X,Y)
        predict = clf.predict(X)
        print(score, math.sqrt(((predict-Y)**2).mean()))
        # Tracer()()
        return score, predict
