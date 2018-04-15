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
from tqdm import tqdm

from utils import layers
from utils import metrics
from utils.nuclei_image import read_image, read_mask, resize_image, rle_seg_map
from utils.nuclei_image_data_flow import RescalePadNucleiImageReader

FIXED_IMAGE_SIZE = 256

SEG_MAP_PATH_PREFIX = 'data/prediction_%s_%s.json'

# TODO do not hard code fixed image size
def predict_sliding_window(padded_image, image_shape, model):
    padded_height, padded_width, _ = padded_image.shape

    h_num_wins = int(padded_height / FIXED_IMAGE_SIZE)
    w_num_wins = int(padded_width / FIXED_IMAGE_SIZE)

    combined_prediction = np.zeros((padded_height, padded_width), dtype=np.float64)

    for i in range(0, h_num_wins):
        for j in range(0, w_num_wins):
            window = padded_image[i * FIXED_IMAGE_SIZE: (i + 1) * FIXED_IMAGE_SIZE, j * FIXED_IMAGE_SIZE: (j + 1) * FIXED_IMAGE_SIZE, :]
            window = np.expand_dims(window, 0)
            prediction = model.predict(window)[0, :, :, 0]
            combined_prediction[i * FIXED_IMAGE_SIZE: (i + 1) * FIXED_IMAGE_SIZE, j * FIXED_IMAGE_SIZE: (j + 1) * FIXED_IMAGE_SIZE] = prediction

    height, width = image_shape

    return combined_prediction[0:height, 0:width]


def read_padded_image(image_path):
    img = read_image(image_path, dehaze=args.dehaze)

    height, width, _ = img.shape

    f_height = int(np.ceil(float(height) / 256))
    f_width = int(np.ceil(float(width) / 256))

    pad_width = [(0, f_height * 256 - height), (0, f_width * 256 - width), (0, 0)]
    img = pad(img, pad_width, mode="reflect")

    return img


def read_image_shape(img_path):
    return imread(img_path).shape[0:2]


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

    print("reading test nuclei images...")
    test_df['image'] = test_df.image_path.map(read_padded_image)

    print("reading test nuclei image shapes...")
    test_df['image_shape'] = test_df.image_path.map(read_image_shape)

    # load model
    print('loading model %s...' % model_path)
    custom_objs = metrics.custom_metrics_dict()
    custom_objs.update(layers.custom_layers_dict())
    print(custom_objs)
    model = load_model(model_path,
                       custom_objects=custom_objs)

    # predict segmentation map
    print('predicting segmentation map...')
    # seg_maps = model.predict(np.array(test_df['image'].tolist()), verbose=1)

    predictions = []

    for i, r in tqdm(test_df.iterrows()):
        prediction = predict_sliding_window(r['image'], r['image_shape'], model)
        predictions.append(prediction)

    # seg_maps = [m[:, :, 0] for m in list(seg_maps)]

    test_df['seg_map'] = pd.Series(predictions, index=test_df.index)
    # test_df['seg_map'] = test_df.apply(resize_segmentation_map, axis=1)
    # test_df['seg_map'] = test_df['seg_map'].map(lambda x: x[0])
    test_df['seg_map'] = test_df['seg_map'].map(rle_seg_map)

    # saves predicted segmentation map
    print('saving predictions...')
    test_df[['id', 'seg_map', 'image_path']]\
        .to_json(SEG_MAP_PATH_PREFIX % (basename(test_data_path), basename(model_path)))

    print("prediction finished")
