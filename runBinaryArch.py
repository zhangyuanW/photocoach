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

featGroups = [['baseline'],['hueComposition','pedestrian','lines'],['bow']]
svmParam = [None,None, {'kernel':'rbf', 'degree':1, 'C':100, 'class_weight':'balanced'}]
voteClassifier = True

normalize = 'std'
# normalize = 'maxmin'
# normalize = 'none'

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
    truth = []
    for sps, (train_idx, test_idx) in enumerate(kf.split(list(range(len(labels))))):
        print ('*****split {0}*****'.format(sps))
        train_label = [labels[i] for i in train_idx]
        #test_img = [labels[i] for i in test_idx]
        test_label = [labels[i] for i in test_idx]
        
        groupFeatdf = getGroupFeat(names, train_idx, test_idx)
        
        df = pd.concat((individualFeatdf,groupFeatdf),axis=1)
        # df = individualFeatdf
        # df = groupFeatdf        
        
        if not voteClassifier:
            featGroups[:] = [list(df.columns.values)]
            svmParam = [None]
        tmp = []
        for feats,params in zip(featGroups,svmParam):
            trainX = np.array([np.concatenate(d) for d in df[feats].loc[train_idx].values])
            testX = np.array([np.concatenate(d) for d in df[feats].loc[test_idx].values])
            
            if normalize != 'none':
                center = np.mean(trainX,axis=0)
                if normalize == 'maxmin':
                    scale = np.max(trainX,axis=0) - np.min(trainX,axis=0)
                else:
                    scale = np.std(trainX,axis=0)
                trainX = np.divide(np.subtract(trainX,center),scale+1e-6)
                testX = np.divide(np.subtract(testX,center),scale+1e-6)
            
            model = svmTrain(trainX, train_label,params)
            
            accu, pred = svmTest(model,testX, test_label, feats)
            tmp.append(pred)
        truth += test_label
        res.append(np.array(tmp))
    res = np.concatenate(res,axis=1)
    if voteClassifier:
        weight = np.array([1,0.8,0.5])
        final_res = weight.dot(res)>1.5
    else:
        final_res = res[0]
    truth = np.array(truth)
    print ('mean accuracy '+ str(np.mean(final_res==truth)))
    return final_res, truth
        
    

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
