import os

from datetime import datetime
from time import time
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape
from keras.models import Model

from image_data_augmentation import BasicNucleiImageReader
from nuclei_sequence import NucleiSequence

TRAIN_DATA_PATH = 'data/train_data.json'
MODEL_DIR = 'models'


def create_model():
    inputs = Input((256, 256, 3))

    c1 = Conv2D(filters=64,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='valid')(inputs)

    u1 = Conv2DTranspose(filters=1,
                         kernel_size=(3, 3),
                         activation='sigmoid')(c1)

    u1 = Reshape(target_shape=(256, 256))(u1)

    return Model(inputs=[inputs], outputs=u1)


def compile_model(model):
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print(model.summary())


def train_model(model):
    train_df = pd.read_json(TRAIN_DATA_PATH)

    image_reader = BasicNucleiImageReader(fixed_img_height=256,
                                          fixed_img_width=256,
                                          fixed_chann_num=3)

    nuclei_image_seq = NucleiSequence(df=train_df,
                                      batch_size=32,
                                      img_reader=image_reader)

    model_name = datetime.fromtimestamp(time()).strftime('%m-%d-%H-%M-%S')
    model_save_path = os.path.join(MODEL_DIR, '%s.h5' % model_name)
    checkpoint = ModelCheckpoint(model_save_path, verbose=1)

    model.fit_generator(generator=nuclei_image_seq,
                        steps_per_epoch=1,
                        epochs=1,
                        max_queue_size=10,
                        workers=2,
                        use_multiprocessing=True,
                        shuffle=True,
                        callbacks=[checkpoint])


if __name__ == '__main__':
    model = create_model()
    compile_model(model)
    train_model(model)

