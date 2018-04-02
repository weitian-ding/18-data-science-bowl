from multiprocessing import Manager

import numpy as np
import pandas as pd
from keras.utils import Sequence
import random

from skimage.transform import rotate

from utils.nuclei_image import random_crop, read_image, read_mask, rescale, FIXED_CHANN_NUM


class BaseNucleiImageReader(object):
    def __init__(self, w=10, q=5, fixed_img_height=None, fixed_img_width=None, border_erosion=False):
        self.fixed_img_height = fixed_img_height
        self.fixed_img_width = fixed_img_width
        self.w = w
        self.q = q
        self.border_erosion = border_erosion

    def __call__(self, _row):
        raise NotImplementedError


class RescaledNucleiImageReader(BaseNucleiImageReader):

    def __init__(self, **kwargs):
        super(RescaledNucleiImageReader, self).__init__(**kwargs)

    def __call__(self, _row):
        img = read_image(img_path=_row['image_path'])
        mask = read_mask(mask_paths=_row['mask_paths'],
                         w=self.w,
                         q=self.q,
                         border_erosion=self.border_erosion)

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
        print('new sequence instantiated...')
        manager = Manager()
        self.cache = manager.dict()

    def __len__(self):
        return int(np.ceil(self.df.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        start_row_idx = idx * self.batch_size
        end_row_idx = start_row_idx + self.batch_size
        row_idx = [i % self.df.shape[0] for i in range(start_row_idx, end_row_idx)]

        # batch data
        batch_df = self.df.iloc[row_idx]
        batch_df_idx = batch_df.index.tolist()

        # load images and masks into memory if not loaded
        missed_idx = set(batch_df_idx) - set(self.cache.keys())
        #print('missed_idx=%s' % str(missed_idx))
        #print('cache=%s' % str(self.cache.keys()))
        #print('%s cache hits, %s cache misses' % (len(batch_df_idx) - len(missed_idx), len(missed_idx)))
        new_data = batch_df.loc[missed_idx].apply(self.img_reader, axis=1)

        # cache
        for i, r in new_data.iterrows():
            self.cache[i] = (r['image'], r['mask'])

        images = []
        masks = []

        # load images and masks from cache
        for idx in batch_df_idx:
            image, mask = self.cache[idx]

            # random rotation
            angle = random.choice([0, 180])
            image = rotate(image, angle)
            mask = rotate(mask, angle)

            images.append(image)
            masks.append(mask)

        return np.array(images), np.array(masks)

    def on_epoch_end(self):
        # shuffles the training set
        self.df = self.df.sample(frac=1)