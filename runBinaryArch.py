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
import util_IO
from util_IO import getFeatLabel, readImageLabel, getIndividualFeat, getGroupFeat
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import chi2_kernel, additive_chi2_kernel
import pandas as pd

# featGroups = [['baseline'],['hueComposition','pedestrian','lines'],['bow']]
featGroups = [['baseline'],['hueComposition','pedestrian','lines']]
svmParam = [None,None, {'kernel':chi2_kernel, 'degree':1, 'C':100, 'class_weight':'balanced'}]
voteClassifier = False
mergeFeats = False

normalize = 'std'
# normalize = 'maxmin'
normalize = 'none'
list_normalize = ['std','std','none']

task_opts = {"binary":('/HighQuality/architecture/','/LowQuality/architecture/'),
            "rotate20":('/HighQuality/architecture/','/HighQuality/architecture/rotate20/'),
            "rotateRandom":('/HighQuality/architecture/','/HighQuality/architecture/rotateRandom/'),
            "rotateRegression":('/HighQuality/architecture/','/HighQuality/architecture/rotateRegression/','dict.pickle')}

def main(data_dir = 'C:/PhotoQualityDataset/', recalc = False, task = 'binary'):
    """
        Read images & label, shuffle, calc individual features, calc group features, train classifier, test classifier
    """
    # read images and labels
    util_IO.imgSubPath = task_opts[task]
    util_IO.task = task
    if task != 'rotateRegression':
        regression = False
        _, labels, names = readImageLabel(data_dir, name_only=True)
        
        # shuffle
        labels, names = shuffle(labels,names, random_state=0)
            
        # 5 fold
        kf = KFold(n_splits=5)
        
        splits = kf.split(list(range(len(labels))))
    else:
        regression = True
        splits, labels, names = readImageLabel(data_dir, name_only=True)
    
    # calc individual features
    individualFeatdf = getIndividualFeat(names, recalc = recalc)
    
    res = []
    truth = []
    for sps, (train_idx, test_idx) in enumerate(splits):
        print ('*****split {0}*****'.format(sps))
        train_label = [labels[i] for i in train_idx]
        #test_img = [labels[i] for i in test_idx]
        test_label = [labels[i] for i in test_idx]
                
        # groupFeatdf = getGroupFeat(names, train_idx, test_idx)
        # df = pd.concat((individualFeatdf,groupFeatdf),axis=1)
        df = individualFeatdf
        # df = groupFeatdf        
        
        if not voteClassifier and mergeFeats:
            featGroups[:] = [list(df.columns.values)]
            svmParam[:] = [None]
        tmp = []
        for feats,params,normType in zip(featGroups,svmParam,list_normalize):
            trainX = np.array([np.concatenate(d) for d in df[feats].loc[train_idx].values])
            testX = np.array([np.concatenate(d) for d in df[feats].loc[test_idx].values])
            
            if normType != 'none':
                center = np.mean(trainX,axis=0)
                if normType == 'maxmin':
                    scale = np.max(trainX,axis=0) - np.min(trainX,axis=0)
                else:
                    scale = np.std(trainX,axis=0)
                trainX = np.divide(np.subtract(trainX,center),scale+1e-6)
                testX = np.divide(np.subtract(testX,center),scale+1e-6)
            model = svmTrain(trainX, train_label,params,regression)
            
            accu, pred = svmTest(model,testX, test_label, feats, regression)
            tmp.append(pred)
        truth += test_label
        res.append(np.array(tmp))
    truth = np.array(truth)
    res = np.concatenate(res,axis=1)
    if voteClassifier:
        weight = np.array([0.6,2,0.8][:len(featGroups)])
        final_res = weight.dot(res)>1
        print ('mean accuracy '+ str(np.mean(final_res==truth)))
    else:
        for feats, partres in zip(featGroups, res):
            print ('mean accuracy '+str(feats)+' :'+ str(np.mean(partres==truth)))
        final_res = res
    
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
    p.add_argument('--task',help="what type of task",type=str,choices=task_opts.keys(),default = 'binary')
    args = p.parse_args()
    main(args.data_dir,args.recalc,args.task)
