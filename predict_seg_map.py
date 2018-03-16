import argparse
from functools import partial

import pandas as pd
import numpy as np
from keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
from os.path import basename
import keras.backend as K

from utils import layers
from utils import metrics
from utils.nuclei_image import read_image, read_mask, rescale

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

    def read_image_fixed_size (image_path):
        img = read_image(image_path)
        return rescale(img, (256, 256, 3))

    test_df['image'] = test_df.image_path.map(read_image_fixed_size)

    # load model
    print('loading model %s...' % model_path)
    custom_objs = metrics.custom_metrics_dict()
    custom_objs.update(layers.custom_layers_dict())
    print(custom_objs)
    model = load_model(model_path,
                       custom_objects=custom_objs)

    # predict segmentation map
    print('predicting segmentation map...')
    seg_maps = model.predict(np.array(test_df['image'].tolist()), verbose=1)
    seg_maps = [m[:, :, 0] for m in list(seg_maps)]
    test_df['seg_map'] = pd.Series(seg_maps, index=test_df.index)
    test_df['seg_map'] = test_df.apply(resize_segmentation_map, axis=1)
    test_df['seg_map'] = test_df['seg_map'].map(lambda x: x[0])

    # saves predicted segmentation map
    print('saving predictions...')
    test_df.to_json(SEG_MAP_PATH_PREFIX % (basename(test_data_path), basename(model_path)))
