"""
    read images, get features and labels
"""
from featureExtractors import calcFeatures, calcModel, runModel, calcPreFeature,featureToUse, groupFeatureToUse
import glob
from scipy.misc import imread
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
import os 
from sklearn.model_selection import KFold
import pickle

imgSubPath = ('/HighQuality/architecture/','/LowQuality/architecture/')
task = 'binary'

def getFeatLabel(data_dir, num=None, featureNames = []):
    """
        Deprecated.
        
        get feature and labels. Shuffle is done.
        
        Args:
            data_dir: path to dataset
            num: if specified, return num good image features and num bad image features
            featureNames: if not specified, use all features in featureExtractors.featureMap
            
        Returns:
            X: N by M feature matrix
            Y: N by 1 label
            names: list of path to the images, in the order of feature
    """
    PATH = [data_dir+imgSubPath[0],data_dir+imgSubPath[1]]
    labels = []
    feats = []
    names = []
    # read both high and low, calc features
    for p,label in zip(PATH,[1,0]):
        filelist = glob.glob(p+'*.jpg')
        if not num:
            num = len(filelist)
        if not filelist:
            raise Exception ('file not found under '+p)
        for i,f in enumerate(filelist):
            feats.append(calcFeatures(imread(f),featureNames))
            names.append(f)
            if i%100 == 0:
                print('Feature extraction: {0} out of {1} images done for label {2}.'.format(i, num, label))
            if num and i >= num-1:
                break
        print('Feature extraction: {0} out of {1} images done for label {2}.'.format(i, num, label))
        labels += [label]*(i+1)
    feats = np.vstack(feats)
    labels = np.array(labels)
    
    # shuffle
    X,Y,names = shuffle(feats,labels, names,random_state=0)
    
    return X,Y,names

def getIndividualFeat(fileLists, num=None, featureNames = [], recalc = False, save_dir = './temp/'):
    """
        Calc Individual Feature for images. If the feature is already calculated and stored (save_dir/feat.npy), load it. Load and store only happens when the whole dataset is used.
        
        Args:
            data_dir: path to dataset
            num: if specified, return num good image features and num bad image features
            featureNames: if not specified, use all features in featureExtractors.featureMap
            
        Returns:
            pd.DataFrame: rows are samples, columns are features
    """
    feats = {}
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not num:
        num = len(fileLists)
    else:
        num = min(num, len(fileLists))
    if not num:
        raise Exception ('No images')
    if not featureNames:
        featureNames = featureToUse
    for feat in featureNames:
        feats[feat] = []
        # if stored feature file exists
        if os.path.exists(save_dir+task+'_{0}.npy'.format(feat)) and not recalc and num == len(fileLists):
            feats[feat] = list(np.load(save_dir+task+'_{0}.npy'.format(feat)))
            print(feat+' read from file done')
        else:
            for i in range(num):
                feats[feat].append(calcFeatures(imread(fileLists[i]),[feat]))
                if i%100 == 0:
                    print('Feature extraction {2}: {0} out of {1} images done.'.format(i, num, feat))
            if num == len(fileLists):
                np.save(save_dir+task+'_{0}.npy'.format(feat), feats[feat])
        print('Feature extraction {2}: {0} out of {1} images done.'.format(num, num, feat))
    feats = pd.DataFrame(feats)
    
    # shuffle
#    X,Y,names = shuffle(feats,labels, names,random_state=0)
    
    return feats
    
def getGroupFeat(fileLists, train_idx, test_idx, featureName = '', save_dir='./temp/', recalc = False):
    """
        Calc features with training images, model it and get 
    """
    featureName = featureName or groupFeatureToUse
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if os.path.exists(save_dir+task+'_{0}.pickle'.format(featureName)) and not recalc:
        print (featureName+' read from '+save_dir+task+'_{0}.pickle'.format(featureName))
        return pd.read_pickle(save_dir+task+'_{0}.pickle'.format(featureName))
    if os.path.exists(save_dir+task+'_pre_{0}.npy'.format(featureName)) and not recalc:
        prefeatures = np.load(save_dir+task+'_pre_{0}.npy'.format(featureName))
        print (featureName+' raw read from '+save_dir+task+'_pre_{0}.npy'.format(featureName))
    else:
        prefeatures = calcPreFeature(fileLists, featureName)
        np.save(save_dir+task+'_pre_{0}.npy'.format(featureName), prefeatures)
    models = calcModel([prefeatures[i] for i in train_idx], featureName)
    features = runModel(prefeatures, models, featureName)
    features = pd.DataFrame({(featureName or groupFeatureToUse): features})
    features.to_pickle(save_dir+task+'_{0}.pickle'.format(featureName))
    return features
    
def readImageLabel(data_dir, num=None, save_dir = './temp/', name_only = True):
    PATH = [data_dir+imgSubPath[0],data_dir+imgSubPath[1]]
    print PATH
    
    if name_only and len(imgSubPath) == 3:
        label = pickle.load(open(data_dir+imgSubPath[1]+imgSubPath[2],'r'))
        nameGroup = [label[k]['name'] for k in label]
        labelGroup = [label[k]['label'] for k in label]
        nameGroup,labelGroup = shuffle(nameGroup,labelGroup, random_state=0)
        kf = KFold(n_splits=5)
        train_labels = []
        train_names = []
        test_labels = []
        test_names = []
        res_tr = []
        res_te = []
        for train_idx,test_idx in kf.split(nameGroup):
            tmp = []
            for tr in train_idx:
                tmp += list(range(5*tr,5*tr+5))
            tmp = shuffle(tmp,random_state=0)
            res_tr.append(tmp)
            tmp = []
            for te in test_idx:
                tmp += list(range(5*te,5*te+5))
            tmp = shuffle(tmp,random_state=0)
            res_te.append(tmp)
        
        labels = []
        names = []
        for ng,lg in zip(nameGroup,labelGroup):
            labels += lg
            names += [data_dir+imgSubPath[1]+nng for nng in ng]
        
        return [(tr,te) for tr,te in zip(res_tr,res_te)], labels, names
        

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if name_only:
        files_high = glob.glob(PATH[0]+'*.jpg')
        files_low = glob.glob(PATH[1]+'*.jpg')
        return [], [1]*len(files_high)+[0]*len(files_low), files_high+files_low 
    if os.path.exists(save_dir+'raw.npy') and num is None:
        images, labels, names = np.load(save_dir+'raw.npy')
        return images, labels, names
    

    labels = []
    images = []
    names = []
    
    # read both high and low, calc features
    for p,label in zip(PATH,[1,0]):
        filelist = glob.glob(p+'*.jpg')
        if not filelist:
            raise Exception ('file not found under '+p)
        for i,f in enumerate(filelist):
            images.append(imread(f))
            names.append(f)
            if i%100 == 0:
                print('Image read: {0} out of {1} images done for label {2}.'.format(i, len(filelist), label))
            if num and i >= num-1:
                break
        labels += [label]*(i+1)
        print('Image read: {0} out of {1} images done for label {2}.'.format(i, len(filelist), label))
    # if num is None:
        # np.save(save_dir+'raw.npy', (images, labels, names))
    return images, labels, names