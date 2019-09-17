"""Learner for distributed RL.
Requires env variables:
  SYMPH_PS_FRONTEND_HOST
  SYMPH_PS_FRONTEND_PORT
  SYMPH_PARAMETER_PUBLISH_PORT
  SYMPH_SPEC_HOST
  SYMPH_SPEC_PORT
"""

from __future__ import absolute_import, division, print_function

import os
from threading import Thread

import liaison.utils as U
import tensorflow as tf
from caraml.zmq import (ZmqClient, ZmqProxyThread, ZmqPub, ZmqServer, ZmqSub,
                        ZmqTimeoutError)
from liaison.distributed import (LearnerDataPrefetcher, ParameterClient,
                                 ParameterPublisher, Trajectory)
from liaison.session.tracker import PeriodicTracker
from liaison.utils import ConfigDict, logging
from queue import Queue
from tensorflow.contrib.framework import nest


class Learner(object):

  # learner does the following operations.
  # instantiates the agent to train.
  # Periodically publishes the agent weights to parameter server
  # Fetches experience data for training

  def __init__(self,
               agent_class,
               agent_config,
               batch_size,
               traj_length,
               seed,
               loggers,
               system_loggers,
               agent_scope='learner',
               prefetch_batch_size=1,
               max_prefetch_queue=1,
               max_fetch_queue=1,
               max_preprocess_queue=1,
               prefetch_processes=1,
               use_gpu=True,
               publish_every=1,
               **session_config):
    """
    Args:
      session_config: Learner config
      agent_class: Agent class to invoke
      agent_config: Agent config to pass to
      batch_size: batch_size
      traj_length: trajectory length to expect from experience
      seed: seed,
      loggers: Log training metrics
      system_loggers: Log system metrics
      agent_scope: Agent scope to initialize tf variables within
      prefetch_batch_size: # of batches to fetch for every request to replay
      max_prefetch_queue: Max-size of the prefetch queue.
      max_preprocess_queue: Max-size of the preprocess queue.
      prefetch_process: # of processes to run in parallel for prefetching.
    """
    self.config = ConfigDict(**session_config)
    self._prefetch_batch_size = prefetch_batch_size
    self._max_prefetch_queue = max_prefetch_queue
    self._max_preprocess_queue = max_preprocess_queue
    self._max_fetch_queue = max_fetch_queue
    self._prefetch_processes = prefetch_processes
    self._loggers = loggers
    self._system_loggers = system_loggers
    self._batch_size = batch_size
    self._traj_length = traj_length

    self._agent_scope = agent_scope
    self._setup_ps_publisher()
    self._setup_ps_client_handle()
    self._setup_exp_fetcher()
    self._setup_spec_client()
    self._get_specs()

    self._step_number = 0
    self._publish_queue = Queue()
    self._publish_thread = Thread(target=self._publish)
    self._publish_thread.start()
    self._publish_tracker = PeriodicTracker(publish_every)

    self._graph = tf.Graph()
    with self._graph.as_default():
      if use_gpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
      else:
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

      self._agent = agent_class(name=agent_scope,
                                action_spec=self._action_spec,
                                seed=seed,
                                **agent_config)

      self.sess.run(tf.global_variables_initializer())

      self._mk_phs(self._traj_spec)
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
                            name='learner/' + spec.name.replace(':', '_') +
                            '_ph')

    self._traj_phs = nest.map_structure(mk_ph, traj_spec)

  def _get_specs(self):
    while True:
      try:
        self._traj_spec, self._action_spec = self.spec_client.request(
            (self._batch_size, self._traj_length))
      except ZmqTimeoutError:
        logging.info('ZmQ timed out for the spec server.')
        continue
      break

  def _setup_spec_client(self):
    self.spec_client = ZmqClient(host=os.environ['SYMPH_SPEC_HOST'],
                                 port=os.environ['SYMPH_SPEC_PORT'],
                                 serializer=U.pickle_serialize,
                                 deserializer=U.pickle_deserialize,
                                 timeout=4)

  def _setup_exp_fetcher(self):
    self._exp_fetcher = LearnerDataPrefetcher(
        batch_size=self._prefetch_batch_size,
        max_prefetch_queue=self._max_prefetch_queue,
        max_fetch_queue=self._max_fetch_queue,
        max_preprocess_queue=self._max_preprocess_queue,
        prefetch_processes=self._prefetch_processes,
        worker_preprocess=None,
        main_preprocess=None)
    self._exp_fetcher.start()

  def _setup_ps_client_handle(self):
    """Initialize self._ps_client and connect it to the ps."""
    self._ps_client = ParameterClient(
        host=os.environ['SYMPH_PS_FRONTEND_HOST'],
        port=os.environ['SYMPH_PS_FRONTEND_PORT'],
        agent_scope=self._agent_scope)

  def _setup_ps_publisher(self):
    self._ps_publisher = ParameterPublisher(
        port=os.environ['SYMPH_PARAMETER_PUBLISH_PORT'],
        agent_scope=self._agent_scope)

  def _initial_publish(self):
    self._publish_variables()
    # blocks until connection is successful.
    self._ps_client.fetch_info()
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
      self._ps_publisher.publish(*data)

  def _get_next_exp(self):
    """Generates iterator to fetch next experience batch."""
    while True:
      batches = self._exp_fetcher.get()
      for batch in batches:
        yield batch

  def main(self):
    exp_iter = self._get_next_exp()
    for _ in range(self.config.n_train_steps):
      additional_system_logs = dict()
      with U.Timer() as batch_timer:
        batch = next(exp_iter)

      feed_dict = {
          ph: val
          for ph, val in zip(nest.flatten(self._traj_phs), nest.flatten(batch))
      }
      with U.Timer() as step_timer:
        log_vals = self._agent.update(self.sess, feed_dict)

      if self._publish_tracker.track_increment():
        with U.Timer() as publish_timer:
          self._publish_variables()

        additional_system_logs = dict(
            publish_time_sec=publish_timer.to_seconds())

      for logger in self._loggers:
        logger.write(log_vals)

      for logger in self._system_loggers:
        logger.write(
            dict(sps=self._batch_size * self._traj_length /
                 float(step_timer.to_seconds()),
                 per_step_time_sec=step_timer.to_seconds(),
                 batch_fetch_time_sec=batch_timer.to_seconds(),
                 **additional_system_logs))

      self._step_number += 1

    self._publish_queue.put(None)  # exit the thread once training ends.
