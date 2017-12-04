"""
    Basic SVM classifier.
    
    Classifier should support cross-validation and return stats including validation accuracy etc.

    Probably we can support train, test function later.
"""
import sklearn.svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import numpy as np

args = {'CUHKPQ': {'kernel':'rbf', 'degree':1, 'C':1, 'class_weight':'balanced'}
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
    
def svmTrain(X,Y):
    """
        train svm model and return model
    """
    clf = sklearn.svm.SVC(**args['CUHKPQ'])
    clf.fit(X, Y) 
    return clf

def svmTest(clf,X,Y):
    """
        test svm model, return accuracy
    """
    predict = clf.predict(X)
    print confusion_matrix(Y,predict)
    return np.mean(predict==Y)