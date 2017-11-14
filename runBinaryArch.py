"""
    Script to run binary classification on CUHKPQ, architecture category
    
    Usage:
        To run from console, run `python runBinaryArch.py --data_dir PATH_TO_PhotoQualityDataset`
"""

import os
import numpy as np
import glob
from buildClassifiers import runBinaryCV
import argparse
from util_IO import getFeatLabel

def main(data_dir='C:/PhotoQualityDataset/'):
    """
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
    main(p.parse_args().data_dir)
