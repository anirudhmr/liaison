"""MLP based model."""

import numpy as np

import sonnet as snt
from liaison.agents.models.utils import *
from liaison.specs import BoundedArraySpec
from sonnet.python.ops import initializers

MINF = np.float32(-1e9)


class Model:

  def __init__(self, hidden_layer_sizes, seed, action_spec, activation='relu'):
    assert action_spec is not None
    assert isinstance(action_spec, BoundedArraySpec)

    self.n_actions = action_spec.maximum - action_spec.minimum + 1
    self.hidden_layer_sizes = hidden_layer_sizes
    self.activation = activation
    self.seed = seed

    with tf.variable_scope('policy_network'):
      self.policy = snt.nets.MLP(
          hidden_layer_sizes + [self.n_actions],
          initializers=dict(w=glorot_uniform(seed),
                            b=initializers.init_ops.Constant(
                                0.1)),  # small bias initializer.
          activate_final=False,
          activation=get_activation_from_str(activation),
      )

    with tf.variable_scope('value_network'):
      self.value = snt.nets.MLP(
          self.hidden_layer_sizes + [1],
          initializers=dict(w=glorot_uniform(self.seed),
                            b=initializers.init_ops.Constant(
                                0.1)),  # small bias initializer.
          activate_final=False,
          activation=get_activation_from_str(self.activation),
      )

  def _dummy_state(self, bs):
    return tf.fill(tf.expand_dims(bs, 0), 0)

  def get_initial_state(self, bs):
    return self._dummy_state(bs)

  def get_logits_and_next_state(self, step_type, _, obs, __):

    if 'features' not in obs:
      raise Exception('features not found in observation.')

    logits = self.policy(obs['features'])
    bs = tf.shape(step_type)[0]
    if 'mask' in obs:
      mask = obs['mask']
      logits = tf.reshape(logits, tf.shape(mask))
      # mask some of the logits
      logits = tf.where(tf.equal(mask, 1), logits,
                        tf.fill(tf.shape(mask), MINF))
    return logits, self._dummy_state(bs), {}

  def get_value(self, _, __, obs, ___):

    if 'features' not in obs:
      raise Exception('features not found in observation.')

    return self.value(obs['features'])
