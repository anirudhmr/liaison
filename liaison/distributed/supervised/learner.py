"""Learner for distributed online Supervised Learning.
Requires env variables:
  SYMPH_PS_PUBLISHING_HOST
  SYMPH_PS_PUBLISHING_PORT
  SYMPH_PS_SERVING_HOST
  SYMPH_PS_SERVING_PORT
  SYMPH_SPEC_HOST
  SYMPH_SPEC_PORT
  SYMPH_IRS_HOST
  SYMPH_IRS_PORT
"""

from __future__ import absolute_import, division, print_function

import copy
import logging
import os
import uuid
from queue import Queue
from threading import Thread

import liaison.utils as U
import tensorflow as tf
from caraml.zmq import (ZmqClient, ZmqFileUploader, ZmqProxyThread, ZmqPub,
                        ZmqServer, ZmqSub, ZmqTimeoutError)
from liaison.distributed import (LearnerDataPrefetcher, ParameterClient,
                                 SimpleParameterPublisher, Trajectory)
from liaison.session.tracker import PeriodicTracker
from liaison.utils import ConfigDict, logging
from tensorflow.contrib.framework import nest
from tensorflow.python.client import timeline

TEMP_FOLDER = '/tmp/liaison/'


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
               use_gpu=True,
               publish_every=1,
               checkpoint_every=100,
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

    self._publish_queue = Queue()
    self._publish_thread = Thread(target=self._publish)
    self._publish_thread.start()
    self._publish_tracker = PeriodicTracker(publish_every)
    self._profile_step = self.config.profile_step
    self._checkpoint_every = checkpoint_every

    self._graph = tf.Graph()
    with self._graph.as_default():
      self._global_step_op = tf.train.get_or_create_global_step(self._graph)

      self._agent = agent_class(name=agent_scope,
                                action_spec=self._action_spec,
                                seed=seed,
                                **agent_config)

      self._mk_phs(self._traj_spec)
      traj_phs = self._traj_phs
      self._agent.build_update_ops(
          step_types=traj_phs['step_type'],
          prev_states=traj_phs['step_output']['next_state'],
          step_outputs=ConfigDict(traj_phs['step_output']),
          observations=copy.copy(traj_phs['observation']),
          rewards=traj_phs['reward'],
          discounts=traj_phs['discount'])

      if use_gpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.intra_op_parallelism_threads = 1
        # config.inter_op_parallelism_threads = 1
        self.sess = tf.Session(config=config)
      else:
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

      self.sess.run(tf.global_variables_initializer())
      self.sess.run(tf.local_variables_initializer())
      self._saver = tf.train.Saver()
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
        logging.info('ZmQ timed out for the spec server. Retrying...')
        continue
      break

  def _setup_spec_client(self):
    self.spec_client = ZmqClient(host=os.environ['SYMPH_SPEC_HOST'],
                                 port=os.environ['SYMPH_SPEC_PORT'],
                                 serializer=U.pickle_serialize,
                                 deserializer=U.pickle_deserialize,
                                 timeout=4)

  def _batch_and_preprocess_trajs(self, l):
    traj = Trajectory.batch(l, self._traj_spec)
    # feed and overwrite the trajectory
    traj['step_output'], traj['step_output']['next_state'], traj[
        'step_type'], traj['reward'], traj['observation'], traj[
            'discount'] = self._agent.update_preprocess(
                step_outputs=ConfigDict(traj['step_output']),
                prev_states=traj['step_output']['next_state'],
                step_types=traj['step_type'],
                rewards=traj['reward'],
                observations=traj['observation'],
                discounts=traj['discount'])
    return traj

  def _setup_exp_fetcher(self):
    config = self.config
    bs = self._batch_size
    # set prefetch_batch_size equal to batch_size / N:
    # split prefetching into N chunks where batch_size % N == 0
    # set N to be as high as possible with an upper limit of 8
    # If batch_size is a prime number, then this will end up
    # prefetching the batch in single chunk
    pf_bs = bs // max([i for i in range(1, 9) if bs % i == 0])

    self._exp_fetcher = LearnerDataPrefetcher(
        batch_size=bs,
        prefetch_batch_size=pf_bs,
        combine_trajs=self._batch_and_preprocess_trajs,
        max_prefetch_queue=config.max_prefetch_queue * pf_bs,
        prefetch_processes=config.prefetch_processes,
        prefetch_threads_per_process=config.prefetch_threads_per_process,
        tmp_dir=config.inmem_tmp_dir)
    self._exp_fetcher.start()

  def _setup_ps_client_handle(self):
    """Initialize self._ps_client and connect it to the ps."""
    self._ps_client = ParameterClient(host=os.environ['SYMPH_PS_SERVING_HOST'],
                                      port=os.environ['SYMPH_PS_SERVING_PORT'],
                                      agent_scope=self._agent_scope)

  def _setup_ps_publisher(self):
    self._ps_publisher = SimpleParameterPublisher(
        host=os.environ['SYMPH_PS_PUBLISHING_HOST'],
        port=os.environ['SYMPH_PS_PUBLISHING_PORT'],
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
    self._publish_queue.put((self.global_step, var_dict))

  def _publish(self):
    while True:
      data = self._publish_queue.get()
      if data is None:
        return
      self._ps_publisher.publish(*data)

  def _get_file_uploader(self):
    return ZmqFileUploader(host=os.environ['SYMPH_IRS_HOST'],
                           port=os.environ['SYMPH_IRS_PORT'],
                           serializer='pyarrow',
                           deserializer='pyarrow')

  def _send_metagraph(self):
    fname = os.path.join(TEMP_FOLDER, str(uuid.uuid4()), 'learner.meta')
    with self._graph.as_default():
      tf.train.export_meta_graph(filename=fname)

    file_uploader = self._get_file_uploader()
    file_uploader.send('register_metagraph',
                       src_fname=fname,
                       dst_fname='learner.meta',
                       dst_dir_name='')
    U.f_remove(fname)

  def _create_ckpt(self):
    # save the model
    export_path = os.path.join(TEMP_FOLDER, str(uuid.uuid4()))
    U.f_mkdir(export_path)
    logging.info('Using %s folder for checkpointing ' % export_path)
    with self._graph.as_default():
      self._saver.save(self.sess,
                       export_path + '/learner',
                       global_step=self.global_step,
                       write_meta_graph=False)

    file_uploader = self._get_file_uploader()
    for fname in os.listdir(export_path):
      if fname.endswith('.meta'):
        continue
      file_uploader.send('register_checkpoint',
                         src_fname=os.path.join(export_path, fname),
                         dst_fname=fname,
                         dst_dir_name='%d/' % self.global_step)
    U.f_remove(export_path)

  def _save_profile(self, options, run_metadata):
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    export_path = os.path.join(TEMP_FOLDER, str(uuid.uuid4()))
    U.f_mkdir(export_path)
    with open(os.path.join(export_path, 'timeline.json'), 'w') as f:
      f.write(ctf)
    file_uploader = self._get_file_uploader()
    file_uploader.send('register_profile',
                       src_fname=os.path.join(export_path, 'timeline.json'),
                       dst_fname='timeline.json',
                       dst_dir_name='')  # dst_dir_name is unused.
    U.f_remove(export_path)

  @property
  def global_step(self):
    return self.sess.run(self._global_step_op)

  def main(self):
    for _ in range(self.config.n_train_steps):
      system_logs = dict()

      # fetch the next training batch
      with U.Timer() as batch_timer:
        batch = self._exp_fetcher.get()

      with U.Timer() as step_timer:
        # run update step on the sampled batch
        feed_dict = {
            ph: val
            for ph, val in zip(nest.flatten(self._traj_phs), nest.flatten(
                batch))
        }
        profile_kwargs = {}
        if self.global_step == self._profile_step:
          profile_kwargs = dict(
              options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
              run_metadata=tf.RunMetadata())

        log_vals = self._agent.update(self.sess, feed_dict, profile_kwargs)

        if profile_kwargs:
          self._save_profile(**profile_kwargs)

      with U.Timer() as log_timer:
        for logger in self._loggers:
          logger.write(log_vals)

      # after first sess.run finishes send the metagraph.
      if self.global_step == 1:
        self._send_metagraph()

      # publish the variables if required.
      if self._publish_tracker.track_increment():
        with U.Timer() as publish_timer:
          self._publish_variables()
        system_logs['publish_time_sec'] = publish_timer.to_seconds()

      # Checkpoint if required
      if self.global_step % self._checkpoint_every == 0:
        with U.Timer() as ckpt_timer:
          self._create_ckpt()
        system_logs['ckpt_time_sec'] = ckpt_timer.to_seconds()

      with U.Timer() as system_log_timer:
        # log system profile
        for logger in self._system_loggers:
          logger.write(
              dict(global_step=self.global_step,
                   sps=self._batch_size * self._traj_length /
                   float(step_timer.to_seconds()),
                   per_step_time_sec=step_timer.to_seconds(),
                   batch_fetch_time_sec=batch_timer.to_seconds(),
                   **system_logs))
      system_logs['log_time_sec'] = log_timer.to_seconds(
      ) + system_log_timer.to_seconds()

    self._publish_queue.put(None)  # exit the thread once training ends.
