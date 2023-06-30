from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
import os
import random


class SelfAttention(Layer):
    def __init__(self, r, da, name, **kwargs):
        self.r = r
        self.da = da
        self.scope = name
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Ws1 = self.add_weight(
            name='Ws1' + self.scope,
            shape=(input_shape[2], self.da),
            initializer='glorot_uniform',
            trainable=True
        )
        self.Ws2 = self.add_weight(
            name='Ws2' + self.scope,
            shape=(self.da, self.r),
            initializer='glorot_uniform',
            trainable=True
        )
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        A1 = tf.matmul(inputs, self.Ws1)
        A1 = tf.tanh(A1)
        A_T = tf.nn.softmax(tf.matmul(A1, self.Ws2))
        A = tf.transpose(A_T, perm=[0, 2, 1])
        B = tf.matmul(A, inputs)
        tile_eye = tf.tile(tf.eye(self.r), [tf.shape(inputs)[0], 1])
        tile_eye = tf.reshape(tile_eye, [-1, self.r, self.r])
        AA_T = tf.matmul(A, A_T) - tile_eye
        P = tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))
        return [B, P]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.da, self.r), (input_shape[0],)]
