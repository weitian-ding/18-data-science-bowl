from keras.utils import Sequence

import numpy as np


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