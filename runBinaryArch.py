"""
    Script to run binary classification on CUHKPQ, architecture category
    
    Usage:
        To run from console, run `python runBinaryArch.py --data_dir PATH_TO_PhotoQualityDataset`
"""

import os
import numpy as np
import glob
from buildClassifiers import runBinaryCV
from classifiers.svm import svmTrain, svmTest
import argparse
from util_IO import getFeatLabel, readImageLabel, getIndividualFeat, getGroupFeat
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import pandas as pd

def main(data_dir = 'C:/PhotoQualityDataset/', recalc = False):
    """
        Read images & label, shuffle, calc individual features, calc group features, train classifier, test classifier
    """
    # read images and labels
    _, labels, names = readImageLabel(data_dir, name_only=True)
    
    # shuffle
    labels, names = shuffle(labels,names, random_state=0)
    
    # calc individual features
    individualFeatdf = getIndividualFeat(names, recalc = recalc)
        
    # 5 fold
    kf = KFold(n_splits=5)
    res = []
    for train_idx, test_idx in kf.split(list(range(len(labels)))):
        train_label = [labels[i] for i in train_idx]
        #test_img = [labels[i] for i in test_idx]
        test_label = [labels[i] for i in test_idx]
        
        groupFeatdf = getGroupFeat(names, train_idx, test_idx)
        df = pd.concat((individualFeatdf,groupFeatdf),axis=1)
        # df = groupFeatdf
        trainX = np.array([np.concatenate(d) for d in df.loc[train_idx].values])
        testX = np.array([np.concatenate(d) for d in df.loc[test_idx].values])
        model = svmTrain(trainX, train_label)
        res.append(svmTest(model,testX, test_label))
    print ('mean accuracy '+ str(sum(res)/len(res)))
    return res
        
    

def main_old(data_dir='C:/PhotoQualityDataset/'):
    """
        Deprecated.
        
        main function. can be used to debug in ipython notebook
        
        Args:
            data_dir: string. Path to the unzipped PhotoQualityDataset folder
    """
    X,Y,_ = getFeatLabel(data_dir)
    
    # run Cross Validation
    runBinaryCV(X,Y)
    
if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='C:/PhotoQualityDataset/', type=str, help='Path to the unzipped PhotoQualityDataset folder')
    p.add_argument('--recalc', help="force recalculate of all features", action="store_true")
    args = p.parse_args()
    main(args.data_dir,args.recalc)
