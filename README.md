# 18-data-science-bowl
[The 2018 Nuclei Image Segmentation Kaggle competition.](https://www.kaggle.com/c/data-science-bowl-2018/)

## Data Preprocessing 
[preprocess.py](preprocess.py): walks the training data directory and testing data directory, read all images into
 a training dataframe and a tesing dataframe respectively.
 
 [train_cv_split.py](train_cv_split.py): splits the training dataframe into a training set and a cross validation set.

## Model Training 
[train.py](train.py): trains and saves a model.

## Prediction
[predict_window.py](predict_window.py): predicts the segmentation maps of abitrary sized test images using a sliding window. 
[predict_rescale.py](predict_window.py): predicts the segmentation maps of rescaled test images. 
