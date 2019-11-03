"""Shell for policy evaluation.

Env variables used:
  SYMPH_PS_FRONTEND_HOST
  SYMPH_PS_FRONTEND_PORT
"""

import os
import tensorflow as tf
from absl import logging
from liaison.specs import ArraySpec
from liaison.utils import ConfigDict
from liaison.env import StepType
from tensorflow.contrib.framework import nest
from liaison.distributed import ParameterClient


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
      batch_size,
      seed,
      # Above args provided by actor.
      agent_class,
      agent_config,
      agent_scope='shell',
      sync_period=None,
      use_gpu=False,
      **kwargs):
    self.config = ConfigDict(kwargs)
    self._obs_spec = obs_spec
    self._sync_checker = SyncEveryNSteps(sync_period)
    assert self._sync_checker.should_sync(0)  # must sync at the beginning
    self._step_number = 0
    self._agent_scope = agent_scope

    self._graph = tf.Graph()
    with self._graph.as_default():
      if use_gpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
      else:
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

      self._agent = agent_class(name=agent_scope,
                                action_spec=action_spec,
                                seed=seed,
                                **agent_config)

      self._batch_size = batch_size
      self._batch_size_ph = tf.placeholder_with_default(
          batch_size, shape=(), name='shell_batch_size_ph')
      self.sess.run(tf.global_variables_initializer())
      self._initial_state_op = self._agent.initial_state(self._batch_size_ph)
      dummy_initial_state = self.sess.run(
          self._initial_state_op)  # weights are random.

      self._mk_phs(dummy_initial_state)
      self._step_output = self._agent.step(self._step_type_ph, self._reward_ph,
                                           self._obs_ph, self._next_state_ph)

      self._variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope=agent_scope)
      self._variable_names = [var.name for var in self._variables]
      logging.info('Number of Variables identified for syncing: %d',
                   len(self._variables))
      logging.info('Variable names for syncing: %s',
                   ', '.join(self._variable_names))
      self._var_name_to_phs = dict()
      self._var_names_to_assign_ops = dict()
      for var in self._variables:
        ph = tf.placeholder(dtype=var.dtype,
                            shape=var.shape,
                            name='assign_%s_ph' % var.name.replace(':', '_'))
        self._var_name_to_phs[var.name] = ph
        self._var_names_to_assign_ops[var.name] = tf.assign(var,
                                                            ph,
                                                            use_locking=True)
    self._setup_ps_client()
    self._next_state = None

  @property
  def next_state(self):
    if self._next_state is None:
      self._next_state = self.sess.run(self._initial_state_op)

    return self._next_state

  def _mk_phs(self, initial_state_dummy_spec):

    def mk_ph(spec):
      return tf.placeholder(dtype=spec.dtype,
                            shape=spec.shape,
                            name='shell_' + spec.name + '_ph')

    self._step_type_ph = tf.placeholder(dtype=tf.int8,
                                        shape=(None, ),
                                        name='shell_step_type_ph')
    self._reward_ph = tf.placeholder(dtype=tf.float32,
                                     shape=(None, ),
                                     name='shell_reward_ph')
    self._obs_ph = nest.map_structure(mk_ph, self._obs_spec)
    self._next_state_ph = tf.placeholder(dtype=initial_state_dummy_spec.dtype,
                                         shape=initial_state_dummy_spec.shape,
                                         name='next_state_ph')

  def _setup_ps_client(self):
    """Initialize self._ps_client and connect it to the ps."""
    self._ps_client = ParameterClient(
        host=os.environ['SYMPH_PS_FRONTEND_HOST'],
        port=os.environ['SYMPH_PS_FRONTEND_PORT'],
        agent_scope=self._agent_scope,
        timeout=self.config.ps_client_timeout,
        not_ready_sleep=self.config.ps_client_not_ready_sleep)

  def _pull_vars(self):
    """get weights from the parameter server."""
    params, unused_info = self._ps_client.fetch_parameter_with_info(
        self._variable_names)
    return params

  def _sync_variables(self):
    var_vals = self._pull_vars()
    if var_vals:
      assert sorted(var_vals.keys()) == sorted(self._variable_names) == sorted(
          self._var_names_to_assign_ops.keys())
      for var_name, assign_op in self._var_names_to_assign_ops.items():
        self.sess.run(
            assign_op,
            feed_dict={self._var_name_to_phs[var_name]: var_vals[var_name]})
      logging.info("Synced weights.")

  def step(self, step_type, reward, observation):
    if self._sync_checker.should_sync(self._step_number):
      self._sync_variables()

    # bass the batch through pre-processing
    step_type, reward, obs, next_state = self._agent.step_preprocess(
        step_type, reward, observation, next_state)
    nest.assert_same_structure(self._obs_ph, observation)
    obs_feed_dict = {
        obs_ph: obs_val
        for obs_ph, obs_val in zip(nest.flatten(self._obs_ph),
                                   nest.flatten(observation))
    }

    step_output = self.sess.run(self._step_output,
                                feed_dict={
                                    self._step_type_ph: step_type,
                                    self._reward_ph: reward,
                                    self._batch_size_ph: self._batch_size,
                                    self._next_state_ph: self.next_state,
                                    **obs_feed_dict,
                                })
    if self._step_number % 100 == 0:
      print(step_output)
    self._next_state = step_output.next_state
    self._step_number += 1
    return step_output

  def step_output_spec(self):

    def mk_spec(tensor):
      return ArraySpec(dtype=tensor.dtype.as_numpy_dtype,
                       shape=tensor.shape,
                       name=tensor.name)

    return dict(nest.map_structure(mk_spec, self._step_output)._asdict())
