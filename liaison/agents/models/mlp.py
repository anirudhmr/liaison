"""MLP based model."""

import sonnet as snt
from sonnet.python.ops import initializers
from liaison.agents.models.utils import *
# from shapeguard import ShapeGuard


class Model:

  def __init__(self, hidden_layer_sizes, n_actions, seed, activation='relu'):
    self.hidden_layer_sizes = hidden_layer_sizes
    self.n_actions = n_actions
    self.activation = activation
    self.seed = seed

    with tf.variable_scope('policy_network'):
      self.policy = snt.nets.MLP(
          hidden_layer_sizes + [n_actions],
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
    return logits, self._dummy_state(bs)

  def get_value(self, _, __, obs, ___):

    if 'features' not in obs:
      raise Exception('features not found in observation.')

    return self.value(obs['features'])
