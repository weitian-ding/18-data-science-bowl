import argparse
import os
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.utils import plot_model

from img_seg_model.unet import Unet
from utils.image_data_augmentation import BasicNucleiImageReader
from utils.metrics import dice_coef_loss
from utils.nuclei_sequence import NucleiSequence

TRAIN_DATA_PATH = 'data/train_data_train_split.json'
CV_DATA_PATH = 'data/train_data_cv_split.json'
MODEL_DIR = 'models'
MODEL_NAME_PREFIX = 'unet'
TB_DIR = 'tensorboard'


def compile_model(model):

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[dice_coef_loss, 'mse', 'acc'])

    model.summary()


def load_cv_data(img_reader):

    print('loading cross validation data...')

    cv_df = pd.read_json(CV_DATA_PATH)
    cv_df = cv_df.apply(img_reader, axis=1)

    print('%s rows of cross validation data loaded' % cv_df.shape[0])

    return np.array(cv_df['image'].tolist()), \
           np.array(cv_df['mask'].tolist())


def train_model(model, batch_size,
                steps_per_epoch,
                epochs,
                max_queue_size,
                workers,
                disable_multi_proc):

    train_df = pd.read_json(TRAIN_DATA_PATH)

    image_reader = BasicNucleiImageReader(fixed_img_height=256,
                                          fixed_img_width=256,
                                          fixed_chann_num=3)

    # create training data sequence
    nuclei_image_seq = NucleiSequence(df=train_df,
                                      batch_size=batch_size,
                                      img_reader=image_reader)

    # load cross validation data
    X_cv, y_cv = load_cv_data(image_reader)

    # create model checkpoint
    timestamp = datetime.fromtimestamp(time()).strftime('%m-%d-%H-%M-%S')
    model_name = '%s_%s.h5' % (timestamp, MODEL_NAME_PREFIX)
    model_save_path = os.path.join(MODEL_DIR, model_name)
    checkpoint = ModelCheckpoint(model_save_path, verbose=1, save_best_only=True)

    # tensorboard
    tb = TensorBoard(log_dir=os.path.join(TB_DIR, model_name),
                     histogram_freq=1,
                     write_graph=False,
                     write_grads=True,
                     write_images=True)

    # early stopping
    earlystop = EarlyStopping(patience=5, verbose=1)

    hist = model.fit_generator(generator=nuclei_image_seq,
                               steps_per_epoch=steps_per_epoch,
                               epochs=epochs,
                               max_queue_size=max_queue_size,
                               workers=workers,
                               use_multiprocessing=(not disable_multi_proc),
                               shuffle=True,
                               validation_data=(X_cv, y_cv),
                               callbacks=[checkpoint, tb, earlystop])

    print('printing training history...')


if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train and save a model.')
    parser.add_argument('--batch-size', action='store', type=int, default=32)
    parser.add_argument('--steps-per-epoch', action='store', type=int, default=20)
    parser.add_argument('--epochs', action='store', type=int, default=50)
    parser.add_argument('--workers', action='store', type=int, default=2)
    parser.add_argument('--max-queue-size', action='store', type=int, default=5)
    parser.add_argument('--disable-multiprocessing', action='store_true', default=False)
    parser.add_argument('--plot-model', action='store_true', default=False)

    args = parser.parse_args()
    print('configs: %s' % args)

    # instantiate model
    model = Unet().build_model()

    # plot model
    if args.plot_model:
        timestamp = datetime.fromtimestamp(time()).strftime('%m-%d-%H-%M-%S')
        model_plot_file = os.path.join(MODEL_DIR, '%s_%s.png' % (MODEL_NAME_PREFIX, timestamp))
        plot_model(model,
                   to_file=model_plot_file,
                   show_shapes=True,
                   show_layer_names=True,
                   rankdir='TB')

    # compile model
    compile_model(model)

    # train model
    train_model(model, epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
                batch_size=args.batch_size,
                workers=args.workers,
                max_queue_size=args.max_queue_size,
                disable_multi_proc=args.disable_multiprocessing)
