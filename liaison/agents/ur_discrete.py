"""Agent that chooses an action uniform randomly from discrete set of choices
at each step."""

from __future__ import absolute_import, division, print_function

import numpy as np

import tensorflow as tf
from absl import logging
from liaison.agents import BaseAgent, StepOutput
from liaison.agents.utils import *

_ACTION_SPEC_UNBOUNDED = 'action spec `{}` is not correctly bounded.'
_DTYPE_NOT_INTEGRAL = '`dtype` must be integral, got {}.'
_SHAPE_NOT_SPECIFIED = 'action spec `shape` must be fully specified, got {}.'


class Agent(BaseAgent):
  """Randomly samples an action based on a discrete BoundedArraySpec."""

  def __init__(self, name, action_spec, seed, **kwargs):

    self._validate_action_spec(action_spec)
    self.seed = seed
    self._name = name
    self._action_spec = action_spec

    class DummyModel:

      def __init__(self):
        pass

    self._model = DummyModel()
    self.set_seed(self.seed)

  def _validate_action_spec(self, action_spec):
    """Check if it's discrete and bounded."""
    if np.issubdtype(action_spec.dtype, np.integer):
      pass
    else:
      raise ValueError(_DTYPE_NOT_INTEGRAL.format(action_spec.dtype))

    if hasattr(action_spec, 'minimum') and hasattr(action_spec, 'maximum'):
      pass
    else:
      raise ValueError(_ACTION_SPEC_UNBOUNDED.format(action_spec))

    if action_spec.shape:
      for dim in action_spec.shape:
        if dim is None:
          raise ValueError(_SHAPE_NOT_SPECIFIED.format(action_spec.shape))
    else:
      raise ValueError(_SHAPE_NOT_SPECIFIED.format(action_spec.shape))

  def _dummy_state(self, batch_size):
    return tf.fill(tf.expand_dims(batch_size, 0), 0)

  def initial_state(self, batch_size):
    return self._dummy_state(batch_size)

  def step(self, step_type, reward, obs, prev_state):
    """Pick a random discrete action from action_spec."""
    with tf.variable_scope(self._name):
      with tf.name_scope('ur_step'):
        batch_size = tf.shape(step_type)[0]
        if 'mask' in obs:
          logits = tf.cast(tf.identity(obs['mask']), tf.float32)
          logits *= 1e9  # multiply by infinity
          action = sample_from_logits(logits, self.seed)
        else:
          base = tf.random.uniform(self._action_spec.shape,
                                   dtype=tf.float32,
                                   minval=0,
                                   maxval=1)

          L = self._action_spec.minimum
          R = self._action_spec.maximum

          action = tf.cast(L + (base * (R - L)), self._action_spec.dtype)
          logits = tf.fill(tf.expand_dims(batch_size, 0), 0)
        return StepOutput(action, logits, self._dummy_state(batch_size))

  def build_update_ops(self, *args, **kwargs):
    global_step = tf.train.get_or_create_global_step()
    self._incr_op = tf.assign(global_step, global_step + 1)

  def update(self, sess, feed_dict):
    gs = sess.run(self._incr_op)
    return {'steps/global_step': gs}
