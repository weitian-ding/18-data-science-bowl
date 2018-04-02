from random import randint

import numpy as np
from scipy import ndimage
from skimage import img_as_float
from skimage.io import imread
from skimage.transform import resize
from scipy.ndimage.morphology import binary_erosion

from utils.DarkChannelRecover import getRecoverScene

FIXED_CHANN_NUM = 3


def read_mask(mask_paths, border_erosion=False, w=10, q=5):
    # read individual mask
    masks = [img_as_float(imread(p)) for p in mask_paths]
    if border_erosion:
        erosion_masks = [binary_erosion(m, border_value=1).astype(np.float32) for m in masks]

    masks = np.stack(masks)

    # merge individual mask
    if border_erosion:
        mask = np.amax(np.stack(erosion_masks), axis=0)
    else:
        mask = np.amax(masks, axis=0)

    # computes the weight based on object boundaries
    distances = np.array([ndimage.distance_transform_edt(m == 0) for m in masks])
    shortest_dist = np.sort(distances, axis=0)
    d1 = shortest_dist[0]
    d2 = shortest_dist[1] if shortest_dist.shape[0] > 1 else np.zeros(d1.shape)
    weight = w * np.exp(-(d1 + d2) ** 2 / (2 * q ** 2))
    weight = 1 + (mask == 0) * weight

    return np.stack([mask, weight], axis=2)


def read_image(img_path):
    img = imread(img_path)[:, :, 0:FIXED_CHANN_NUM]
    img = getRecoverScene(img, refine=True)
    return img_as_float(img)


def random_crop(img_path, mask_paths, fixed_img_height, fixed_img_width):
    img = read_image(img_path)
    mask = read_mask(mask_paths)
    masked_img = np.dstack((img, mask))

    height, width, chann = img.shape

    h_start_idx = randint(0, height - fixed_img_height)
    w_start_idx = randint(0, width - fixed_img_width)

    cropped = masked_img[h_start_idx:h_start_idx + fixed_img_height,
              w_start_idx:w_start_idx + fixed_img_width, :]

    return cropped[:, :, 0:chann], cropped[:, :, chann:]


def rescale(img, shape):
    return resize(img, shape, mode='constant', preserve_range=False)





