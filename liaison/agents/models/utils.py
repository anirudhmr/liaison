import sonnet as snt
import tensorflow as tf
from sonnet.python.ops import initializers


def glorot_uniform(seed, scale=1.0):
  # See https://sonnet.readthedocs.io/en/latest/api.html#variancescaling
  # Also see https://github.com/keras-team/keras/issues/52
  f = initializers.init_ops.VarianceScaling
  # f = snt.initializers.VarianceScaling
  return f(scale=scale, seed=seed, mode='fan_avg', distribution='uniform')


def glorot_normal(seed):
  # See https://sonnet.readthedocs.io/en/latest/api.html#variancescaling
  # Also see https://github.com/keras-team/keras/issues/52
  f = initializers.init_ops.VarianceScaling
  # f = snt.initializers.VarianceScaling
  return f(scale=1.0,
           seed=seed,
           mode='fan_avg',
           distribution='truncated_normal')


def get_activation_from_str(activation):
  activation = activation.lower()
  if activation == 'relu':
    return tf.nn.relu
  elif activation == 'leaky_relu':
    return tf.nn.leaky_relu
