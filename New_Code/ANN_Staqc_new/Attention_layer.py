import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

# 设置随机种子
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('注意力层应该接受一个包含2个输入的列表。')
        if not input_shape[0][2] == input_shape[1][2]:
            raise ValueError('嵌入大小应该相同。')

        # 初始化权重矩阵
        self.kernel = self.add_weight(
            shape=(input_shape[0][2], input_shape[0][2]),
            initializer='glorot_uniform',
            name='kernel',
            trainable=True
        )

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        a = K.dot(inputs[0], self.kernel)
        y_trans = K.permute_dimensions(inputs[1], (0, 2, 1))
        b = K.batch_dot(a, y_trans, axes=[2, 1])
        return K.tanh(b)

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], input_shape[1][1])
