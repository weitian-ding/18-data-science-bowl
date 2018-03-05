import os
from glob import glob

import pandas as pd

INPUT_DIR = 'data'
OUTPUT_DIR = 'data'


def list_all_images():
    image_paths = glob(os.path.join(INPUT_DIR, 'stage1_*', '*', '*', '*'))

    def parse_img_path(img_path):
        _, _group, _id, _type, _ = img_path.split('/')
        _stage, _split = _group.split('_')
        return pd.Series({'id': _id, 'type': _type, 'stage': _stage, 'split': _split})

    img_df = pd.DataFrame({'path': image_paths})
    img_df = img_df.merge(img_df['path'].apply(parse_img_path), left_index=True, right_index=True)

    print(img_df.head())

    return img_df


def preprocess(df, labeled=True):
    X = df.loc[df['type'] == 'images']
    X = X[['id', 'path']]
    X.rename(columns={'path': 'image_path'}, inplace=True)

    if not labeled:
        return X

    y = df.loc[df['type'] == 'masks']\
        .groupby('id')\
        .agg({'path': (lambda paths: list(paths))})\
        .reset_index()\
        .rename(columns={'path': 'mask_paths'}, inplace=False)

    return pd.merge(X, y, left_on='id', right_on='id')


if __name__ == '__main__':
    print('loading all images in %s...' % INPUT_DIR)
    img_df = list_all_images()
    print('%s images found' % img_df.shape[0])

    train_df = img_df.loc[img_df['split'] == 'train']
    test_df = img_df.loc[img_df['split'] == 'test']

    train_df = preprocess(train_df)
    print('preprocessing training data...')
    print(train_df.head())
    train_df.to_json(os.path.join(OUTPUT_DIR, 'train_data.json'))
    print('%s rows in training data' % train_df.shape[0])

    test_df = preprocess(test_df, labeled=False)
    print('preprocessing testing data...')
    print(test_df.head())
    test_df.to_json(os.path.join(OUTPUT_DIR, 'test_data.json'))
    print('%s rows in testing data' % test_df.shape[0])
