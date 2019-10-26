import numpy as np
import tensorflow as tf
from absl import logging
from liaison.agents import BaseAgent
from liaison.agents import StepOutput

_ACTION_SPEC_UNBOUNDED = 'action spec `{}` is not correctly bounded.'
_DTYPE_NOT_INTEGRAL = '`dtype` must be integral, got {}.'
_SHAPE_NOT_SPECIFIED = 'action spec `shape` must be fully specified, got {}.'


class Agent(BaseAgent):

  def __init__(self, name, action_spec, seed, model=None, **kwargs):

    self.set_seed(seed)
    self._name = name
    self._action_spec = action_spec
    self._load_model(model or {})

  def initial_state(self, bs):
    return self._model.initial_state(bs)

  def step(self, step_type, reward, obs, prev_state):
    with tf.variable_scope(self._name):
      logits, next_state = self._model.get_logits_and_next_state(
          step_type, reward, obs, prev_state)
      action = self.sample_from_logits(logits, self.seed)
      return StepOutput(action, logits, next_state)

  def build_update_ops(self, *args, **kwargs):
    global_step = tf.train.get_or_create_global_step()
    self.old_policy

  def update(self, sess, feed_dict):
    gs = sess.run(self._incr_op)
    return {'steps/global_step': gs}
