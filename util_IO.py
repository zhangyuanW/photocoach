"""
    read images, get features and labels
"""
from featureExtractors import calcFeatures
import glob
def getFeatLabel(data_dir, num=None):
    """
        get feature and labels. Shuffle is done.
        
        Args:
            data_dir: path to dataset
            num: if specified, return num good image features and num bad image features
            
        Returns:
            X: N by M feature matrix
            Y: N by 1 label
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
            if num and i >= num-1:
                break
        labels += [label]*(i+1)
    feats = np.vstack(feats)
    labels = np.array(labels)
    
    # shuffle
    X,Y = shuffle(feats,labels,random_state=0)
    
    return X,Y