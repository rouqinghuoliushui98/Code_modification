import tensorflow as tf
import numpy as np


class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.size = size if size is not None else None
        self.mode = mode

    def build(self, input_shape):
        if self.size is None or self.mode == 'sum':
            self.size = int(input_shape[-1])
        super(PositionEmbedding, self).build(input_shape)

    def call(self, inputs):
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
        position_j = 1. / \
            tf.pow(10000., 2 * tf.range(self.size /
                   2, dtype=tf.float32) / self.size)
        position_j = tf.expand_dims(position_j, 0)

        position_i = tf.cumsum(tf.ones_like(inputs[:, :, 0]), 1) - 1
        position_i = tf.expand_dims(position_i, 2)
        position_ij = tf.matmul(position_i, position_j)

        position_ij_2i = tf.sin(position_ij)[..., tf.newaxis]
        position_ij_2i_1 = tf.cos(position_ij)[..., tf.newaxis]
        position_ij = tf.concat([position_ij_2i, position_ij_2i_1], axis=-1)
        position_ij = tf.reshape(position_ij, (batch_size, seq_len, self.size))

        if self.mode == 'sum':
            return position_ij + inputs
        elif self.mode == 'concat':
            return tf.concat([position_ij, inputs], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)
