import os
from datetime import datetime
from time import time

import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model

from image_data_augmentation import BasicNucleiImageReader
from img_seg_model.unet import Unet
from metrics import dice_coef_loss
from nuclei_sequence import NucleiSequence

STEPS_PER_EPOCH = 20

TRAIN_DATA_PATH = 'data/train_data.json'
MODEL_DIR = 'models'
MODEL_NAME_PREFIX = 'unet'
TB_DIR = 'tensorboard'


def compile_model(model):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[dice_coef_loss, 'mse', 'acc'])

    print(model.summary())


def train_model(model):
    train_df = pd.read_json(TRAIN_DATA_PATH)

    image_reader = BasicNucleiImageReader(fixed_img_height=256,
                                          fixed_img_width=256,
                                          fixed_chann_num=3)

    nuclei_image_seq = NucleiSequence(df=train_df,
                                      batch_size=32,
                                      img_reader=image_reader)

    timestamp = datetime.fromtimestamp(time()).strftime('%m-%d-%H-%M-%S')
    model_save_path = os.path.join(MODEL_DIR, '%s_%s.h5' % (timestamp, MODEL_NAME_PREFIX))
    checkpoint = ModelCheckpoint(model_save_path, verbose=1)

    tb = TensorBoard(log_dir=TB_DIR,
                     histogram_freq=STEPS_PER_EPOCH,
                     write_graph=True,
                     write_grads=True,
                     write_images=True)

    model.fit_generator(generator=nuclei_image_seq,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=20,
                        max_queue_size=5,
                        workers=2,
                        use_multiprocessing=True,
                        shuffle=True,
                        callbacks=[checkpoint, tb])


if __name__ == '__main__':
    model = Unet().build_model()

    # plot model
    timestamp = datetime.fromtimestamp(time()).strftime('%m-%d-%H-%M-%S')
    model_plot_file = os.path.join(MODEL_DIR, '%s_%s.png' % (MODEL_NAME_PREFIX, timestamp))
    plot_model(model,
               to_file=model_plot_file,
               show_shapes=True,
               show_layer_names=True,
               rankdir='TB')

    compile_model(model)
    train_model(model)

