"""Learner for distributed RL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from liaison.utils import logging
from liaison.utils import ConfigDict
from liaison.distributed import Trajectory
import tensorflow as tf
from queue import Queue
from threading import Thread
from tensorflow.contrib.framework import nest


class Learner(object):

  # learner does the following operations.
  # instantiates the agent to train.
  # Periodically publishes the agent weights to parameter server
  # Fetches experience data for training

  def __init__(self,
               session_config,
               agent_class,
               agent_config,
               spec_handle,
               ps_publish_handle,
               ps_client_handle,
               replay_handle,
               batch_size,
               traj_length,
               agent_scope='learner',
               prefetch_batch_size=1,
               use_gpu=True,
               **learner_config):
    """
    Args:
      session_config: Learner config
      agent_class: Agent class to invoke
      agent_config: Agent config to pass to
      actor_handle: Needed to remotely fetch action_spec and obs_scope
      batch_size: batch_size
      ps_publish_handle: handle to parameter publisher.
    """
    self.config = ConfigDict(**learner_config)
    traj_spec = spec_handle.get_traj_spec(batch_size, traj_length)
    action_spec = spec_handle.get_action_spec(batch_size, traj_length)

    self._ps_publish_handle = ps_publish_handle
    self._ps_client_handle = ps_client_handle
    self._replay_handle = replay_handle

    self._step_number = 0
    self._publish_queue = Queue()
    self._publish_thread = Thread(target=self._publish)
    self._publish_thread.start()

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
                                **agent_config)

      self.sess.run(tf.global_variables_initializer())

      self._mk_phs(traj_spec)
      self._agent.build_update_ops(**self._traj_phs)

      self._variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope=agent_scope)
      self._variable_names = [var.name for var in self._variables]
      logging.info('Number of Variables identified for publishing: %d',
                   len(self._variables))
      logging.info('Variable names for publishing: %s',
                   ', '.join(self._variable_names))
      self._initial_publish()

  def _mk_phs(self, traj_spec):

    def mk_ph(spec):
      return tf.placeholder(dtype=spec.dtype,
                            shape=spec.shape,
                            name='learner_' + spec.name + '_ph')

    self._traj_phs = nest.map_structure(mk_ph, traj_spec)

  def _initial_publish(self):
    # blocks until connection is successful.
    self._ps_client_handle.fetch_info()
    self._publish_variables()

  def _publish_variables(self):
    var_vals = self.sess.run(self._variables)
    var_dict = dict()
    for var_name, var_val in zip(self._variable_names, var_vals):
      var_dict[var_name] = var_val
    self._publish_queue.put((self._step_number, var_dict))

  def _publish(self):
    while True:
      data = self._publish_queue.get()
      if data is None:
        return
      self._ps_publish_handle.publish(data)

  def train(self):
    for _ in range(self.config.n_train_steps):
      batch, = self._replay_handle.get()
      feed_dict = {
          ph: val
          for ph, val in zip(nest.flatten(self._traj_phs), nest.flatten(batch))
      }
      log_vals = self._agent.update(self.sess, feed_dict)

      self._step_number += 1
      if self._step_number % self.config.publish_every == 0:
        self._publish_variables()

    self._publish_queue.put(None)  # exit the thread once training ends.
