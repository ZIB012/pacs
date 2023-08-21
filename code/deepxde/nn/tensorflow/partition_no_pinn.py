from .nn import NN
from .random_fnn import partition_random_FNN
from .. import activations
from .. import initializers
from .. import regularizers
from ...backend import tf
import numpy as np

from keras import backend as K
from keras.layers import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, activation, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name="kernel",
            initial_value=w_init(shape=(input_shape[1], self.units),
                                 dtype='float32'), trainable=True)
        
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name="bias",
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True)
        
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.activation(tf.matmul(x, self.w) + self.b)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

