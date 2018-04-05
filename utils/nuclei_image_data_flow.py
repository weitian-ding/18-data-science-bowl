from multiprocessing import Manager

import numpy as np
import pandas as pd
from keras.utils import Sequence
import random

from skimage.transform import rotate
from skimage.util import  pad

from utils.nuclei_image import random_crop, read_image, read_mask, resize_image, FIXED_CHANN_NUM


class BaseNucleiImageReader(object):
    def __init__(self, w=10, q=5, border_erosion=False, dehaze=False):
        self.w = w
        self.q = q
        self.border_erosion = border_erosion
        self.dehaze = dehaze

    def __call__(self, _row):
        raise NotImplementedError


class FixedSizeNucleiImageReader(BaseNucleiImageReader):
    def __init__(self,fixed_img_size=None, **kwargs):
        self.fixed_img_size = fixed_img_size
        super(FixedSizeNucleiImageReader, self).__init__(**kwargs)

    def _read_image_and_mask(self, _row):
        img = read_image(img_path=_row['image_path'], dehaze=self.dehaze)
        mask = read_mask(mask_paths=_row['mask_paths'],
                         w=self.w,
                         q=self.q,
                         border_erosion=self.border_erosion)
        return img, mask


class ResizeNucleiImageReader(FixedSizeNucleiImageReader):

    def __init__(self, **kwargs):
        super(ResizeNucleiImageReader, self).__init__(**kwargs)

    def __call__(self, _row):
        img, mask = self._read_image_and_mask(_row)

        # resize image and mask to fixed_img_height * fixed_img_width
        img = resize_image(img,
                           shape=(self.fixed_img_size, self.fixed_img_size, FIXED_CHANN_NUM))
        mask = resize_image(mask,
                            shape=(self.fixed_img_size, self.fixed_img_size, 2))

        return pd.Series({'image': img, 'mask': mask})


class RescalePadNucleiImageReader(FixedSizeNucleiImageReader):

    def __init__(self, mode, **kwargs):
        super(RescalePadNucleiImageReader, self).__init__(**kwargs)
        self.mode = mode

    def __call__(self, _row):
        img, mask = self._read_image_and_mask(_row)

        # rescale
        rescale_height, rescale_width = RescalePadNucleiImageReader.get_rescaled_shape(img, self.fixed_img_size)

        img = resize_image(img,
                           shape=(rescale_height, rescale_width, FIXED_CHANN_NUM))
        mask = resize_image(mask,
                            shape=(rescale_height, rescale_width, 2))

        # pad the image to a square
        pad_width = [(0, self.fixed_img_size - rescale_height), (0, self.fixed_img_size - rescale_width), (0, 0)]
        img = pad(img, pad_width, mode=self.mode)
        mask = pad(mask, pad_width, mode=self.mode)

        return pd.Series({'image': img, 'mask': mask})

    @staticmethod
    def get_rescaled_shape(img, fixed_img_size):
        height, width, _ = img.shape

        if max(height, width) <= fixed_img_size:
            return height, width

        # Rescale the image so max(height, width) = fixed_img_size
        if height > width:
            rescale_height = fixed_img_size
            rescale_width = int(np.floor(width * (float(fixed_img_size) / height)))
        else:
            rescale_height = int(np.floor(height * (float(fixed_img_size) / width)))
            rescale_width = fixed_img_size

        return rescale_height, rescale_width


# TODO update the code to add another channel in mask
class NucleiImageReader(BaseNucleiImageReader):

    def __init__(self, **kwargs):
        super(NucleiImageReader, self).__init__(**kwargs)

    def __call__(self, _row):
        img = read_image(img_path=_row['image_path'], dehaze=self.dehaze)
        mask = read_mask(mask_paths=_row['mask_paths'],
                         w=self.w,
                         q=self.q,
                         border_erosion=self.border_erosion)

        return pd.Series({'image': img, 'mask': mask})


class NucleiSequence(Sequence):

    def __init__(self, df, batch_size,
                 img_reader,
                 fixed_image_size):
        self.df = df
        self.batch_size = batch_size
        self.img_reader = img_reader
        self.fixed_image_size = fixed_image_size
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
        new_data = batch_df.loc[missed_idx].apply(self.img_reader, axis=1)

        # cache
        for i, r in new_data.iterrows():
            self.cache[i] = (r['image'], r['mask'])

        images = []
        masks = []

        # load images and masks from cache
        for idx in batch_df_idx:
            image, mask = self.cache[idx]

            # random crop
            stacked = np.dstack((image, mask))

            height, width, chann = image.shape

            h_start_idx = random.randint(0, height - self.fixed_image_size)
            w_start_idx = random.randint(0, width - self.fixed_image_size)

            cropped = stacked[h_start_idx:h_start_idx + self.fixed_image_size,
                              w_start_idx:w_start_idx + self.fixed_image_size, :]

            image, mask = (cropped[:, :, 0:chann], cropped[:, :, chann:])


            # random rotation
            angle = random.choice([0, 90, 180, 270])
            image = rotate(image, angle)
            mask = rotate(mask, angle)

            images.append(image)
            masks.append(mask)

        return np.array(images), np.array(masks)

    def on_epoch_end(self):
        # shuffles the training set
        self.df = self.df.sample(frac=1)