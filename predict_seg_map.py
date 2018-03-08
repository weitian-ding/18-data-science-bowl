import argparse
from functools import partial

import pandas as pd
import numpy as np
from keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
from os.path import basename

from utils import metrics
from utils.image_data_augmentation import read_image, read_mask

SEG_MAP_PATH_PREFIX = 'data/seg_map_%s_%s.json'


def resize_segmentation_map(_row):
    shape = imread(_row['image_path']).shape
    resized = resize(_row['seg_map'],
                  (shape[0], shape[1]),
                   mode='constant',
                   preserve_range=False)
    return [resized]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict segmentation maps.')
    parser.add_argument('--model-path', action='store', type=str, required=True)
    parser.add_argument('--test-data-path', action='store', type=str, required=True)

    args = parser.parse_args()
    test_data_path = args.test_data_path
    model_path = args.model_path

    # load test data
    print('loading test data...')
    test_df = pd.read_json(test_data_path)\
        .sort_values('id')\
        .reset_index(drop=True)

    read_image_fixed_size = partial(read_image,
                                    fixed_img_height=256,
                                    fixed_img_width=256,
                                    fixed_chann_num=3)

    test_df['image'] = test_df.image_path.map(read_image_fixed_size)

    # load model
    print('loading model %s...' % model_path)
    model = load_model(model_path,
                       custom_objects=metrics.custom_metrics_dict())

    # predict segmentation map
    print('predicting segmentation map...')
    seg_maps = model.predict(np.array(test_df['image'].tolist()), verbose=1)
    test_df['seg_map'] = pd.Series(list(seg_maps), index=test_df.index)
    test_df['seg_map'] = test_df.apply(resize_segmentation_map, axis=1)
    test_df['seg_map'] = test_df['seg_map'].map(lambda x: x[0])

    # saves predicted segmentation map
    print('saving predictions...')
    test_df.to_json(SEG_MAP_PATH_PREFIX % (basename(test_data_path), basename(model_path)))
