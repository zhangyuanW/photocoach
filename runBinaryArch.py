"""
    Script to run binary classification on CUHKPQ, architecture category
    
    Usage:
        To run from console, run `python runBinaryArch.py --data_dir PATH_TO_PhotoQualityDataset`
"""

import os
import numpy as np
import glob
from sklearn.utils import shuffle
from scipy.misc import imread
from buildClassifiers import runBinaryCV
from featureExtractors import calcFeatures
import argparse


def main(data_dir='C:/PhotoQualityDataset/'):
    """
        main function. can be used to debug in ipython notebook
        
        Args:
            data_dir: string. Path to the unzipped PhotoQualityDataset folder
    """
    PATH = [data_dir+'/HighQuality/architecture/',data_dir+'/LowQuality/architecture/']
    labels = []
    feats = []
    
    # read both high and low, calc features
    for p,label in zip(PATH,[1,0]):
        filelist = glob.glob(p+'*.jpg')
        for i,f in enumerate(filelist):
            feats.append(calcFeatures(imread(f)))
            if i%100 == 0:
                print('Feature extraction: {0} out of {1} images done for label {2}.'.format(i, len(filelist), label))
        labels += [label]*len(filelist)
    feats = np.vstack(feats)
    labels = np.array(labels)
    
    # shuffle
    X,Y = shuffle(feats,labels,random_state=0)
    
    # run Cross Validation
    runBinaryCV(X,Y)
    
if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='C:/PhotoQualityDataset/', type=str, help='Path to the unzipped PhotoQualityDataset folder')
    main(p.parse_args().data_dir)
