# README - photocoach

BY Zhangyuan Wang, Jiaxi Chen, Ziran Zhang

## Usage
* To run binary classification task on high quality vs low quality:
    *   `python runBinaryArch.py --task binary --data_dir ./data/`
* To train for another task (e.g. rotation related), specify different `task` option. You may need to first create the augmented dataset by running script in `testNewDataset.ipynb` 
* By default the code will cache features and read previously calculated ones in `./temp/`. To force recalculation, use the option `--recalc`.  
    
## Architecture
*   Features:
    *   Each featureExtractor should take an image as nparray and return the feature as row vector
    *   Add features to `./features/` folder as separate files
    *   Import the features in `./featureExtractors.py`
    *   Define individual features (features that can be calculated by itself), and group features (features that need all the training sample points to calculate, e.g. result from KMeans). Take a look at `featureExtractors.py`.
*   Classifier:
    *   Add classifiers to `./classifiers/` folder as separate files
    *   Import the classifiers in `./buildClassifiers.py`
*   Put them together:
    *   Check out `runBinaryArch.py` and `utilIO.py` for usage.
    
## Roadmap
*   Complete infrastructures and setup
*   For binary classification of [CUHK-PQ](http://mmlab.ie.cuhk.edu.hk/archive/CUHKPQ/Dataset.htm):
    *   Add featureExtractors
    *   Add classifiers
*   Further steps:
    *   Try the previous features on [Cropping Dataset](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_cuhk_crop_dataset.html)
    *   Add new features and tuning
    *   Construct new dataset by doing random cropping etc.
    *   Apply CNN
    
