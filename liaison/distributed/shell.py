"""Shell for policy evaluation."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from absl import logging
from easydict import EasyDict as ConfigDict


class SyncEveryNSteps:

  def __init__(self, sync_period):
    self.sync_period = sync_period

  def should_sync(self, step):
    return step % self.sync_period == 0


class Shell:
  """
  Shell has the following tasks.

  (1) Create a TF Agent graph.
  (2) Extract the learnable exposed weights from the TF graph.
  (3) Connect to parameter server and sync the weights regularly.
  """

  def __init__(
      self,
      action_spec,
      obs_spec,
      agent_class,
      agent_config,
      ps_handle,
      agent_scope='shell',
      batch_size=1,
      sync_period=None,
      use_gpu=False,
      **kwargs,
  ):
    self.config = ConfigDict(kwargs)
    self._ps_handle = ps_handle
    self._obs_spec = obs_spec
    self._sync_checker = SyncEveryNSteps(sync_period)
    self._step_number = 0

    with tf.Graph().as_default() as self._graph:
      if use_gpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
      else:
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

        self._agent = agent_class(name=agent_scope,
                                  action_spec=action_spec,
                                  **agent_config)

      self._batch_size = batch_size
      self._batch_size_ph = tf.placeholder_with_default(
          batch_size, shape=(), name='shell_batch_size_ph')
      self.sess.run(tf.global_variables_initializer())
      dummy_initial_state = self.sess.run(
          self._agent.initial_state(
              self._batch_size_ph))  # weights are random.

      self._mk_phs(dummy_initial_state)
      self._step_output = self._agent.step(self._step_type_ph, self._reward_ph,
                                           self._obs_ph, self._prev_state_ph)

      self._variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope=agent_scope)
      self._variable_names = [var.name for var in self._variables]
      logging.info('Number of Variables identified for syncing: %d',
                   len(self._variables))
      logging.info('Variable names for syncing: %s',
                   ', '.join(self._variable_names))
      self.sess.run(tf.initialize_all_variables())
      self._var_name_to_phs = dict()
      self._var_names_to_assign_ops = dict()
      for var in self._variables:
        ph = tf.placeholder(dtype=var.dtype,
                            shape=var.shape,
                            name='assign_%s_ph' % var.name)
        self._var_name_to_phs[var.name] = ph
        self._var_names_to_assign_ops[var.name] = tf.assign(var,
                                                            ph,
                                                            use_locking=True)
      self._sync_variables()

  def _mk_phs(self, initial_state_dummy_spec):

    def mk_ph(spec):
      return tf.placeholder(dtype=spec.dtype,
                            shape=spec.shape,
                            name='shell_' + spec.name + '_ph')

    self._step_type_ph = tf.placeholder(dtype=tf.int32,
                                        shape=(None, ),
                                        name='shell_step_type_ph')
    self._reward_ph = tf.placeholder(dtype=tf.float32,
                                     shape=(None, ),
                                     name='shell_reward_ph')
    self._obs_ph = tf.nest.map_structure(mk_ph, self._obs_spec)
    self._prev_state_ph = tf.placeholder(dtype=initial_state_dummy_spec.dtype,
                                         shape=initial_state_dummy_spec.shape,
                                         name='prev_state_ph')

  def _sync_variables(self):
    var_vals = self._ps_handle.pull(self._variable_names)
    assert sorted(var_vals.keys()) == sorted(self._variable_names) == sorted(
        self._var_names_to_assign_ops.keys())
    for var_name, assign_op in self._var_names_to_assign_ops.items():
      self.sess.run(
          assign_op,
          feed_dict={self._var_name_to_phs[var_name]: var_vals[var_name]})
    logging.info("Synced weights.")

  def step(self, step_type, reward, obs):
    if self._sync_checker.should_sync(self._step_number):
      self._sync_variables()

    tf.nest.assert_same_structure(self._obs_ph, obs)
    obs_ph_list = tf.nest.flatten(self._obs_ph)
    obs = tf.nest.flatten(obs)
    assert len(obs) == len(obs_ph_list)
    obs_feed_dict = {
        obs_ph: obs_val
        for obs_ph, obs_val in zip(obs_ph_list, obs)
    }

    step_output = self.sess.run(self._step_output,
                                feed_dict={
                                    self._step_type_ph: step_type,
                                    self._reward_ph: reward,
                                    self._batch_size_ph: self._batch_size,
                                    **obs_feed_dict
                                })
    return step_output
