from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, BatchNormalization, Concatenate, Dropout
from keras.models import Model


class Unet(object):

    def __init__(self):
        # TODO parametrize mdoel configurations
        pass

    def build_model(self):
        inputs = Input((256, 256, 3))
        # normed = BatchNormalization()(inputs)
        normed = BatchNormalization()(inputs)

        c1 = Conv2D(filters=16,
                    kernel_size=(3, 3),
                    activation='elu',
                    kernel_initializer='he_normal',
                    padding='same')(normed)
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
        c5 = Dropout(0.2)(c5)
        c5 = Conv2D(filters=256,
                    kernel_size=(3, 3),
                    activation='elu',
                    kernel_initializer='he_normal',
                    padding='same')(c5)
        # p5 = MaxPooling2D(pool_size=(2, 2))(c5)

        u5 = Conv2DTranspose(filters=128,
                             kernel_size=(2, 2),
                             strides=(2, 2),
                             kernel_initializer='he_normal',
                             activation='elu')(c5)
        u5 = Conv2D(filters=128,
                    kernel_size=(3, 3),
                    padding='same',
                    kernel_initializer='he_normal',
                    activation='elu')(Concatenate()([u5, c4]))
        u5 = Dropout(0.2)(u5)
        u5 = Conv2D(filters=128,
                    kernel_size=(3, 3),
                    padding='same',
                    kernel_initializer='he_normal',
                    activation='elu')(u5)

        u1 = Conv2DTranspose(filters=64,
                             kernel_size=(2, 2),
                             strides=(2, 2),
                             kernel_initializer='he_normal',
                             activation='elu')(u5)
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
        u3 = Conv2D(filters=16,
                    kernel_size=(3, 3),
                    padding='same',
                    kernel_initializer='he_normal',
                    activation='elu')(u3)

        output = Conv2D(filters=1,
                        kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer='he_normal',
                        activation='sigmoid')(u3)

        output = Reshape(target_shape=(256, 256))(output)

        return Model(inputs=[inputs], outputs=output)