"""
MLP based model.
"""

import sonnet as snt
from liaison.agents.models.utils import *
from shapeguard import ShapeGuard


class Model:

  def __init__(self, hidden_layer_sizes, n_actions, seed, activation='relu'):
    self.hidden_layer_sizes = hidden_layer_sizes
    self.n_actions = n_actions
    self.activation = activation
    self.seed = seed
    self.policy = None
    self.value = None

  def _validate_features_shape(self, features):
    sg = ShapeGuard()
    sg.guard(features, "?, *")

  def _dummy_state(self, bs):
    return tf.fill(tf.expand_dims(bs, 0), 0)

  def get_initial_state(self, bs):
    return self._dummy_state(bs)

  def get_logits_and_next_state(self, step_type, _, obs, _):
    with tf.variable_scope('mlp_policy'):
      self.policy = snt.nets.MLP(
          self.hidden_layer_sizes + [self.n_actions],
          w_init=glorot_uniform(self.seed),
          b_init=snt.initializers.Constant(0.1),  # small bias initializer.
          activate_final=False,
          activation=get_activation_from_str(self.activation),
      )
      if 'features' not in obs:
        print('Use MLPEncoder on observation output.')
        raise Exception('features not found in observation.')

      features = obs['features']
      self._validate_features_shape(features)
      logits = self.policy(features)

      bs = tf.shape(step_type)[0]
      return logits, _dummy_state(bs)

  def get_value(self, _, _, obs, _):
    with tf.variable_scope('mlp_value'):
      self.value = snt.nets.MLP(
          self.hidden_layer_sizes + [1],
          w_init=glorot_uniform(self.seed),
          b_init=snt.initializers.Constant(0.1),  # small bias initializer.
          activate_final=False,
          activation=get_activation_from_str(self.activation),
      )
      self._validate_features_shape(obs['features'])
      value_op = self.value(obs['features'])
    return value_op
