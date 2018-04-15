import argparse
from functools import partial

import pandas as pd
import numpy as np
from keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
from skimage.util import pad
from os.path import basename
import keras.backend as K

from utils import layers
from utils import metrics
from utils.nuclei_image import read_image, read_mask, resize_image, rle_seg_map
from utils.nuclei_image_data_flow import RescalePadNucleiImageReader

SEG_MAP_PATH_PREFIX = 'data/seg_map_%s_%s.json'

# TODO do not hard code fixed image size
def resize_segmentation_map(_row):
    img = imread(_row['image_path'])
    prediction = _row['seg_map']

    try:
        height, width, _ = img.shape
    except ValueError:
        print("%s is ill shaped, assuming 2D" % _row['image_path'])
        height, width = img.shape

    rescale_height, rescale_width = RescalePadNucleiImageReader.get_rescaled_shape(img, 256)

    # crop
    prediction = prediction[0:rescale_height, 0:rescale_width]

    # resize
    resized = resize(prediction,
                    (height, width),
                    mode='constant',
                    preserve_range=False)

    return [resized]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict segmentation maps.')
    parser.add_argument('--model-path', action='store', type=str, required=True)
    parser.add_argument('--test-data-path', action='store', type=str, required=True)
    parser.add_argument('--dehaze', action='store_true', default=False)

    args = parser.parse_args()
    test_data_path = args.test_data_path
    model_path = args.model_path

    # load test data
    print('loading test data...')
    test_df = pd.read_json(test_data_path)\
        .sort_values('id')\
        .reset_index(drop=True)

    #TODO do not hardcode image size
    def read_image_fixed_size (image_path):
        img = read_image(image_path, dehaze=args.dehaze)

        # rescale image
        rescale_height, rescale_width = RescalePadNucleiImageReader.get_rescaled_shape(img, 256)
        img = resize_image(img,
                           shape=(rescale_height, rescale_width, 3))

        pad_width = [(0, 256 - rescale_height), (0, 256 - rescale_width), (0, 0)]
        img = pad(img, pad_width, mode="reflect")

        return img

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

    # run length encode
    test_df['seg_map'] = test_df['seg_map'].map(rle_seg_map)

    # saves predicted segmentation map
    print('saving predictions...')
    filename = SEG_MAP_PATH_PREFIX % (basename(test_data_path), basename(model_path))
    test_df[['id', 'seg_map', 'image_path']]\
        .to_json(filename, compression='gzip')

    print("prediction finished, %s" % filename)
