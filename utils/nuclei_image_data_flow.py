import numpy as np
import pandas as pd
from keras.utils import Sequence

from utils.nuclei_image import random_crop, read_image, read_mask, rescale, FIXED_CHANN_NUM


class BaseNucleiImageReader(object):
    def __init__(self, fixed_img_height=None, fixed_img_width=None):
        self.fixed_img_height = fixed_img_height
        self.fixed_img_width = fixed_img_width

    def __call__(self, _row):
        raise NotImplementedError


class RescaledNucleiImageReader(BaseNucleiImageReader):

    def __init__(self, **kwargs):
        super(RescaledNucleiImageReader, self).__init__(**kwargs)

    def __call__(self, _row):
        img = read_image(img_path=_row['image_path'])
        mask = read_mask(mask_paths=_row['mask_paths'])

        img = rescale(img,
                      shape=(self.fixed_img_height, self.fixed_img_width, FIXED_CHANN_NUM))
        mask = rescale(mask,
                       shape=(self.fixed_img_height, self.fixed_img_width, 2))

        return pd.Series({'image': img, 'mask': mask})


# TODO update the code to add another channel in mask
class RandomCropImageReader(BaseNucleiImageReader):

    def __init__(self, **kwargs):
        super(RandomCropImageReader, self).__init__(**kwargs)

    def __call__(self, _row):
        img, mask = random_crop(img_path=_row['image_path'],
                                mask_paths=_row['mask_paths'],
                                fixed_img_height=self.fixed_img_height,
                                fixed_img_width=self.fixed_img_width)

        return pd.Series({'image': img, 'mask': mask})


class NucleiSequence(Sequence):

    def __init__(self, df, batch_size,
                 img_reader):
        self.df = df
        self.batch_size = batch_size
        self.img_reader = img_reader

    def __len__(self):
        return int(np.ceil(self.df.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size
        row_idx = [i % self.df.shape[0] for i in range(start_idx, end_idx)]

        batch_df = self.df.iloc[row_idx]
        batch_df = batch_df.apply(self.img_reader, axis=1)

        return np.array(batch_df['image'].tolist()), \
               np.array(batch_df['mask'].tolist())

    def on_epoch_end(self):
        # shuffles the training set
        self.df = self.df.sample(frac=1)