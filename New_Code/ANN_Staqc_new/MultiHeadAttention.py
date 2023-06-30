import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)


class ScaledDotProductAttention(Layer):
    def __init__(self, return_attention=False, history_only=False, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only
        self.intensity = None
        self.attention = None

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
            'history_only': self.history_only,
        }
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape
        output_shape = query_shape[:-1] + value_shape[-1:]
        if self.return_attention:
            attention_shape = query_shape[:2] + (key_shape[1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        if isinstance(mask, list):
            mask = mask[1]
        feature_dim = K.shape(query)[-1]
        e = K.batch_dot(query, key, axes=2) / \
            K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
        if self.history_only:
            query_len, key_len = K.shape(query)[1], K.shape(key)[1]
            indices = K.expand_dims(K.arange(0, key_len), axis=0)
            upper = K.expand_dims(K.arange(0, query_len), axis=-1)
            e -= 10000.0 * \
                K.expand_dims(K.cast(indices > upper, K.floatx()), axis=0)
        if mask is not None:
            e -= 10000.0 * \
                (1.0 - K.cast(K.expand_dims(mask, axis=-2), K.floatx()))
        self.intensity = e
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        self.attention = e / K.sum(e, axis=-1, keepdims=True)
        v = K.batch_dot(self.attention, value)
        if self.return_attention:
            return [v, self.attention]
        return v


class MultiHeadAttention_(Layer):
    """Multi-head attention layer.
    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 head_num,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.
        :param head_num: Number of heads.
        :param activation: Activations for linear mappings.
        :param use_bias: Whether to use bias term.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        :param history_only: Whether to only use history in attention layer.
        """
        super(MultiHeadAttention_, self).__init__(**kwargs)
        self.supports_masking = True
        self.head_num = head_num
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.history_only = history_only

        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.Wo = None
        self.bq = None
        self.bk = None
        self.bv = None
        self.bo = None

        self.intensity = None
        self.attention = None

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'history_only': self.history_only,
        }
        base_config = super(MultiHeadAttention_, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
            return q[:-1] + (v[-1],)
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask


def build(self, input_shape):
    if isinstance(input_shape, list):
        q, k, v = input_shape
    else:
        q = k = v = input_shape
    feature_dim = int(v[-1])
    if feature_dim % self.head_num != 0:
        raise IndexError('Invalid head number %d with the given input dim %d' % (
            self.head_num, feature_dim))
    self.Wq = self.add_weight(
        shape=(int(q[-1]), feature_dim),
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        name='%s_Wq' % self.name,
    )
    if self.use_bias:
        self.bq = self.add_weight(
            shape=(feature_dim,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            name='%s_bq' % self.name,
        )
    self.Wk = self.add_weight(
        shape=(int(k[-1]), feature_dim),
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        name='%s_Wk' % self.name,
    )
    if self.use_bias:
        self.bk = self.add_weight(
            shape=(feature_dim,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            name='%s_bk' % self.name,
        )
    self.Wv = self.add_weight(
        shape=(int(v[-1]), feature_dim),
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        name='%s_Wv' % self.name,
    )
    if self.use_bias:
        self.bv = self.add_weight(
            shape=(feature_dim,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            name='%s_bv' % self.name,
        )
    self.Wo = self.add_weight(
        shape=(feature_dim, feature_dim),
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        name='%s_Wo' % self.name,
    )
    if self.use_bias:
        self.bo = self.add_weight(
            shape=(feature_dim,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            name='%s_bo' % self.name,
        )
    super(MultiHeadAttention_, self).build(input_shape)


def call(self, inputs, mask=None):
    if isinstance(inputs, list):
        q, k, v = inputs
    else:
        q = k = v = inputs
    if isinstance(mask, list):
        q_mask, k_mask, v_mask = mask
    else:
        q_mask = k_mask = v_mask = mask
    q = tf.matmul(q, self.Wq)
    k = tf.matmul(k, self.Wk)
    v = tf.matmul(v, self.Wv)
    if self.use_bias:
        q += self.bq
        k += self.bk
        v += self.bv
    if self.activation is not None:
        q = self.activation(q)
        k = self.activation(k)
        v = self.activation(v)
    scaled_dot_product_attention = ScaledDotProductAttention(
        history_only=self.history_only,
        name='%s-Attention' % self.name,
    )
    y = scaled_dot_product_attention(
        inputs=[
            self._reshape_to_batches(q, self.head_num),
            self._reshape_to_batches(k, self.head_num),
            self._reshape_to_batches(v, self.head_num),
        ],
        mask=[
            self._reshape_mask(q_mask, self.head_num),
            self._reshape_mask(k_mask, self.head_num),
            self._reshape_mask(v_mask, self.head_num),
        ],
    )
    y = self._reshape_from_batches(y, self.head_num)
    y = tf.matmul(y, self.Wo)
    if self.use_bias:
        y += self.bo
    if self.activation is not None:
        y = self.activation(y)

    # Add shape information to tensor
    input_shape = [tf.shape(q), tf.shape(k), tf.shape(v)]
    output_shape = self.compute_output_shape(input_shape)
    if output_shape[1] is not None:
        output_shape = (-1,) + output_shape[1:]
        y = tf.reshape(y, output_shape)
    return y
