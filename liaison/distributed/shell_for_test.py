# Shell to be used for tests
import copy
import os

import tensorflow as tf
from absl import logging
from liaison.env import StepType
from liaison.specs import ArraySpec
from liaison.utils import ConfigDict
from tensorflow.contrib.framework import nest


class Shell:

  def __init__(
      self,
      action_spec,
      obs_spec,
      seed,
      # Above args provided by actor.
      agent_class,
      agent_config,
      batch_size,
      agent_scope='shell',
      use_gpu=False,
      **kwargs):
    self.config = ConfigDict(kwargs)
    self._obs_spec = obs_spec
    self._step_number = 0
    self._agent_scope = agent_scope

    self._graph = tf.Graph()
    with self._graph.as_default():
      if use_gpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
      else:
        self.sess = tf.Session()

      self._agent = agent_class(name=agent_scope,
                                action_spec=action_spec,
                                seed=seed,
                                **agent_config)

      self._batch_size_ph = tf.placeholder_with_default(batch_size,
                                                        shape=(),
                                                        name='shell_batch_size_ph')
      self._initial_state_op = self._agent.initial_state(self._batch_size_ph)
      dummy_initial_state = self.sess.run(self._initial_state_op)  # weights are random.

      self._mk_phs(dummy_initial_state)
      self._step_output = self._agent.step(self._step_type_ph, self._reward_ph,
                                           copy.copy(self._obs_ph), self._next_state_ph)
      self.sess.run(tf.global_variables_initializer())

      self._variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=agent_scope)
      self._variable_names = [var.name for var in self._variables]
      logging.info('Number of Variables identified for syncing: %d', len(self._variables))
      logging.info('Variable names for syncing: %s', ', '.join(self._variable_names))
      self._var_name_to_phs = dict()
      self._var_names_to_assign_ops = dict()
      for var in self._variables:
        ph = tf.placeholder(dtype=var.dtype,
                            shape=var.shape,
                            name='assign_%s_ph' % var.name.replace(':', '_'))
        self._var_name_to_phs[var.name] = ph
        self._var_names_to_assign_ops[var.name] = tf.assign(var, ph, use_locking=True)
    self._next_state = None

  @property
  def next_state(self):
    if self._next_state is None:
      self._next_state = self.sess.run(self._initial_state_op)

    return self._next_state

  def _mk_phs(self, initial_state_dummy_spec):

    def mk_ph(spec):
      return tf.placeholder(dtype=spec.dtype, shape=spec.shape, name='shell_' + spec.name + '_ph')

    self._step_type_ph = tf.placeholder(dtype=tf.int8, shape=(None, ), name='shell_step_type_ph')
    self._reward_ph = tf.placeholder(dtype=tf.float32, shape=(None, ), name='shell_reward_ph')
    self._obs_ph = nest.map_structure(mk_ph, self._obs_spec)
    self._next_state_ph = tf.placeholder(dtype=initial_state_dummy_spec.dtype,
                                         shape=initial_state_dummy_spec.shape,
                                         name='next_state_ph')

  def step(self, step_type, reward, observation):
    # bass the batch through pre-processing
    step_type, reward, obs, next_state = self._agent.step_preprocess(step_type, reward,
                                                                     observation, self.next_state)
    nest.assert_same_structure(self._obs_ph, observation)
    obs_feed_dict = {
        obs_ph: obs_val
        for obs_ph, obs_val in zip(nest.flatten(self._obs_ph), nest.flatten(observation))
    }

    step_output = self.sess.run(self._step_output,
                                feed_dict={
                                    self._step_type_ph: step_type,
                                    self._reward_ph: reward,
                                    self._next_state_ph: next_state,
                                    **obs_feed_dict,
                                })
    self._next_state = step_output.next_state
    self._step_number += 1
    return step_output

  def step_output_spec(self):

    def mk_spec(tensor):
      return ArraySpec(dtype=tensor.dtype.as_numpy_dtype, shape=tensor.shape, name=tensor.name)

    return dict(nest.map_structure(mk_spec, self._step_output)._asdict())

  def sync(self):
    pass
