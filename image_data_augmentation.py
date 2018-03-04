import pandas as pd
from skimage import img_as_float
from skimage.io import imread
from skimage.transform import resize

import numpy as np

import os


def read_mask(mask_paths,
              fixed_img_height=None,
              fixed_img_width=None):
    if fixed_img_height is not None and fixed_img_width is not None:
        mask_shape = (fixed_img_height, fixed_img_width)
    else:
        mask_shape = None

    mask = None

    for mask_path in mask_paths:
        if not os.path.exists(mask_path):
            print('%s does not exists' % mask_path)

        _mask = imread(mask_path)

        if mask_shape is not None:
            _mask = resize(_mask,
                           mask_shape,
                           mode='constant',
                           preserve_range=True)

        if mask is None:
            mask = np.zeros(mask_shape) if mask_shape is not None else \
                np.zeros(_mask.shape)

        mask = np.maximum(mask, _mask)

    mask = np.ndarray.astype(mask, np.uint8)
    mask = img_as_float(mask)

    return mask


def read_image(img_path,
               fixed_img_height=None,
               fixed_img_width=None,
               fixed_chann_num=None):
    if not os.path.exists(img_path):
        print('%s does not exists' % img_path)

    img = imread(img_path)[:, :, 0:3]

    if fixed_img_height is not None and fixed_img_width is not None and fixed_chann_num is not None:
        img = resize(img,
                     (fixed_img_height, fixed_img_width, fixed_chann_num),
                     mode='constant',
                     preserve_range=True)

    img = np.ndarray.astype(img, np.uint8)
    img = img_as_float(img)

    return img


class BaseNucleiImageReader(object):
    def __init__(self, fixed_img_height, fixed_img_width, fixed_chann_num):
        self.fixed_img_height = fixed_img_height
        self.fixed_img_width = fixed_img_width
        self.fixed_chann_num = fixed_chann_num

    def __call__(self, _row):
        raise NotImplementedError


class BasicNucleiImageReader(BaseNucleiImageReader):

    def __init__(self, **kwargs):
        super(BasicNucleiImageReader, self).__init__(**kwargs)

    def __call__(self, _row):
        img = read_image(img_path=_row['image_path'],
                         fixed_img_height=self.fixed_img_height,
                         fixed_img_width=self.fixed_img_width,
                         fixed_chann_num=self.fixed_chann_num)

        mask = read_mask(mask_paths=_row['mask_paths'],
                         fixed_img_height=self.fixed_img_height,
                         fixed_img_width=self.fixed_img_width)

        return pd.Series({'image': img, 'mask': mask})
