from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
import random


class PositionWiseFeedForward(Layer):
    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.trainable = trainable
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self.inner_dim),
            initializer='glorot_uniform',
            trainable=self.trainable,
            name="weights_inner")
        self.weights_out = self.add_weight(
            shape=(self.inner_dim, self.model_dim),
            initializer='glorot_uniform',
            trainable=self.trainable,
            name="weights_out")
        self.bias_inner = self.add_weight(
            shape=(self.inner_dim,),
            initializer='uniform',
            trainable=self.trainable,
            name="bias_inner")
        self.bias_out = self.add_weight(
            shape=(self.model_dim,),
            initializer='uniform',
            trainable=self.trainable,
            name="bias_out")
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs):
        if tf.dtypes.as_dtype(inputs.dtype) != tf.float32:
            inputs = tf.cast(inputs, tf.float32)
        inner_out = tf.nn.relu(
            tf.matmul(inputs, self.weights_inner) + self.bias_inner)
        outputs = tf.matmul(inner_out, self.weights_out) + self.bias_out
        print("==", outputs.shape)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
