from keras.engine import Layer
import keras.backend as K


class RepeatVector2D(Layer):

    def __init__(self, **kwargs):
        super(RepeatVector2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RepeatVector2D, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.stack([x, x], axis=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 2)


def custom_layers_dict():
    return {
        'RepeatVector2D': RepeatVector2D
    }