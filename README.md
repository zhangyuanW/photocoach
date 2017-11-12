# photocoach

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
    
## Architecture
*   Features:
    *   Each featureExtractor should take an image as nparray and return the feature as row vector (TBD)
    *   Add features to `./features/` folder as separate files
    *   Import the features in `./featureExtractors.py`
*   Classifier:
    *   For most experiments, use cross-validation. So storeing and loading of classifiers may not be necessary. Check the code for more details about interfaces.
    *   Add classifiers to `./classifiers/` folder as separate files
    *   Import the classifiers in `./buildClassifiers.py`
*   Put them together:
    *   For each task, use one script in `./`. This should implement data IO, calling the feature function and classifier, then produce results.
