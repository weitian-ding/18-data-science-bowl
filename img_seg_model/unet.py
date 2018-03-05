from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, BatchNormalization, Concatenate, Dropout
from keras.models import Model


class Unet(object):

    def __init__(self):
        # TODO parametrize mdoel configurations
        pass

    def build_model(self):
        inputs = Input((256, 256, 3))
        # normed = BatchNormalization()(inputs)

        c1 = Conv2D(filters=16,
                    kernel_size=(3, 3),
                    activation='elu',
                    kernel_initializer='he_normal',
                    padding='same')(inputs)
        c1 = Dropout(0.1)(c1)
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
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(filters=32,
                    kernel_size=(3, 3),
                    activation='elu',
                    kernel_initializer='he_normal',
                    padding='same')(c2)
        p2 = MaxPooling2D(pool_size=(2, 2))(c2)

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

        c4 = Conv2D(filters=128,
                    kernel_size=(3, 3),
                    activation='elu',
                    kernel_initializer='he_normal',
                    padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(filters=128,
                    kernel_size=(3, 3),
                    activation='elu',
                    kernel_initializer='he_normal',
                    padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(filters=256,
                    kernel_size=(3, 3),
                    activation='elu',
                    kernel_initializer='he_normal',
                    padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(filters=256,
                    kernel_size=(3, 3),
                    activation='elu',
                    kernel_initializer='he_normal',
                    padding='same')(c5)
        # p5 = MaxPooling2D(pool_size=(2, 2))(c5)

        u6 = Conv2DTranspose(filters=128,
                             kernel_size=(2, 2),
                             strides=(2, 2),
                             padding='same')(c5)
        u6 = Conv2D(filters=128,
                    kernel_size=(3, 3),
                    padding='same',
                    kernel_initializer='he_normal',
                    activation='elu')(Concatenate()([u6, c4]))
        u6 = Dropout(0.2)(u6)
        u6 = Conv2D(filters=128,
                    kernel_size=(3, 3),
                    padding='same',
                    kernel_initializer='he_normal',
                    activation='elu')(u6)

        u7 = Conv2DTranspose(filters=64,
                             kernel_size=(2, 2),
                             strides=(2, 2),
                             padding='same')(u6)
        u7 = Conv2D(filters=64,
                    kernel_size=(3, 3),
                    padding='same',
                    kernel_initializer='he_normal',
                    activation='elu')(Concatenate()([u7, c3]))
        u7 = Dropout(0.2)(u7)
        u7 = Conv2D(filters=64,
                    kernel_size=(3, 3),
                    padding='same',
                    kernel_initializer='he_normal',
                    activation='elu')(u7)

        u8 = Conv2DTranspose(filters=32,
                             kernel_size=(2, 2),
                             strides=(2, 2),
                             padding='same')(u7)
        u8 = Conv2D(filters=32,
                    kernel_size=(3, 3),
                    padding='same',
                    kernel_initializer='he_normal',
                    activation='elu')(Concatenate()([u8, c2]))
        u8 = Dropout(0.1)(u8)
        u8 = Conv2D(filters=32,
                    kernel_size=(3, 3),
                    padding='same',
                    kernel_initializer='he_normal',
                    activation='elu')(u8)

        u9 = Conv2DTranspose(filters=16,
                             kernel_size=(2, 2),
                             strides=(2, 2),
                             padding='same')(u8)
        u9 = Conv2D(filters=16,
                    kernel_size=(3, 3),
                    padding='same',
                    kernel_initializer='he_normal',
                    activation='elu')(Concatenate()([u9, c1]))
        u9 = Dropout(0.1)(u9)
        u9 = Conv2D(filters=16,
                    kernel_size=(3, 3),
                    padding='same',
                    kernel_initializer='he_normal',
                    activation='elu')(u9)

        output = Conv2D(filters=1,
                        kernel_size=(1, 1),
                        activation='sigmoid')(u9)

        output = Reshape(target_shape=(256, 256))(output)

        return Model(inputs=[inputs], outputs=output)