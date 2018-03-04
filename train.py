import os

from datetime import datetime
from time import time
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, BatchNormalization, Concatenate, Dropout
from keras.models import Model

from image_data_augmentation import BasicNucleiImageReader
from nuclei_sequence import NucleiSequence

TRAIN_DATA_PATH = 'data/train_data.json'
MODEL_DIR = 'models'


def create_model():
    inputs = Input((256, 256, 3))
    # normed = BatchNormalization()(inputs)

    c1 = Conv2D(filters=16,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(inputs)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(filters=16,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(filters=32,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(filters=32,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2,2))(c2)

    c3 = Conv2D(filters=64,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(filters=64,
                kernel_size=(3, 3),
                activation='elu',
                kernel_initializer='he_normal',
                padding='same')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    u1 = Conv2DTranspose(filters=64,
                         kernel_size=(2, 2),
                         strides=(2, 2),
                         kernel_initializer='he_normal',
                         activation='elu')(p3)
    u1 = Conv2D(filters=64,
                kernel_size=(3, 3),
                padding='same',
                kernel_initializer='he_normal',
                activation='elu')(Concatenate()([u1, c3]))
    u1 = Dropout(0.2)(u1)
    u1 = Conv2D(filters=64,
                kernel_size=(3, 3),
                padding='same',
                kernel_initializer='he_normal',
                activation='elu')(u1)

    u2 = Conv2DTranspose(filters=32,
                         kernel_size=(2, 2),
                         strides=(2, 2),
                         kernel_initializer='he_normal',
                         activation='elu')(u1)
    u2 = Conv2D(filters=32,
                kernel_size=(3, 3),
                padding='same',
                kernel_initializer='he_normal',
                activation='elu')(Concatenate()([u2, c2]))
    u2 = Dropout(0.2)(u2)
    u2 = Conv2D(filters=32,
                kernel_size=(3, 3),
                padding='same',
                kernel_initializer='he_normal',
                activation='elu')(u2)

    u3 = Conv2DTranspose(filters=16,
                         kernel_size=(2, 2),
                         strides=(2, 2),
                         kernel_initializer='he_normal',
                         activation='elu')(u2)
    u3 = Conv2D(filters=16,
                kernel_size=(3, 3),
                padding='same',
                kernel_initializer='he_normal',
                activation='elu')(Concatenate()([u3, c1]))
    u3 = Dropout(0.2)(u3)
    u3 = Conv2D(filters=1,
                kernel_size=(1, 1),
                padding='same',
                kernel_initializer='he_normal',
                activation='sigmoid')(u3)

    output = Reshape(target_shape=(256, 256))(u3)

    return Model(inputs=[inputs], outputs=output)


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
                        steps_per_epoch=20,
                        epochs=20,
                        max_queue_size=5,
                        workers=2,
                        use_multiprocessing=True,
                        shuffle=True,
                        callbacks=[checkpoint])


if __name__ == '__main__':
    model = create_model()
    compile_model(model)
    train_model(model)

