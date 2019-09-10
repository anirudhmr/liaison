"""TODO(arc): doc_string."""

from __future__ import absolute_import, division, print_function

import os
import time
from threading import Thread

import numpy as np
import tensorflow as tf
from absl import logging
from liaison.distributed import ExpSender, LearnerDataPrefetcher
from liaison.replay import ReplayLoadBalancer, UniformReplay
from liaison.utils import ConfigDict
from tensorplex import Loggerplex, Tensorplex

_LOCALHOST = 'localhost'

SYMPH_COLLECTOR_FRONTEND_PORT = '6000'
SYMPH_SAMPLER_FRONTEND_PORT = '6001'
SYMPH_COLLECTOR_BACKEND_PORT = '6002'
SYMPH_SAMPLER_BACKEND_PORT = '6003'
SYMPH_PREFETCH_QUEUE_PORT = '6004'
SYMPH_LOGGERPLEX_PORT = '6010'
SYMPH_TENSORPLEX_PORT = '6011'

BATCH_SIZE = 4
MAX_REPLAY_SIZE = 100


def run_loggerplex(session_config):
  """
          Launches a loggerplex server.
          It helps distributed logging.
      """
  folder = session_config.folder
  loggerplex_config = session_config.loggerplex

  loggerplex = Loggerplex(os.path.join(folder, 'logs'),
                          level=loggerplex_config.level,
                          overwrite=loggerplex_config.overwrite,
                          show_level=loggerplex_config.show_level,
                          time_format=loggerplex_config.time_format)
  port = SYMPH_LOGGERPLEX_PORT
  loggerplex.start_server(port)


def run_tensorplex(session_config):
  """
          Launches a tensorplex process.
          It receives data from multiple sources and
          send them to tensorboard.
      """
  folder = os.path.join(session_config.folder, 'tensorboard')
  tensorplex_config = session_config.tensorplex

  tensorplex = Tensorplex(
      folder,
      max_processes=tensorplex_config.max_processes,
  )
  """
          Tensorboard categories:
              learner/replay/eval: algorithmic level, e.g. reward, ...
              ***-core: low level metrics, i/o speed, computation time, etc.
              ***-system: Metrics derived from raw metric data in core,
                  i.e. exp_in/exp_out
      """
  (tensorplex.register_normal_group('learner').register_indexed_group(
      'agent', tensorplex_config.agent_bin_size).register_indexed_group(
          'eval', 4).register_indexed_group('replay', 10))

  port = SYMPH_TENSORPLEX_PORT
  tensorplex.start_server(port=port)


class ReplayTest(tf.test.TestCase):

  def _setup_env(self):
    os.environ.update(
        dict(
            SYMPH_COLLECTOR_BACKEND_PORT=SYMPH_COLLECTOR_BACKEND_PORT,
            SYMPH_SAMPLER_BACKEND_PORT=SYMPH_SAMPLER_BACKEND_PORT,
            SYMPH_SAMPLER_FRONTEND_PORT=SYMPH_SAMPLER_FRONTEND_PORT,
            SYMPH_SAMPLER_FRONTEND_HOST=_LOCALHOST,
            SYMPH_COLLECTOR_FRONTEND_PORT=SYMPH_COLLECTOR_FRONTEND_PORT,
            SYMPH_COLLECTOR_FRONTEND_HOST=_LOCALHOST,
            SYMPH_PREFETCH_QUEUE_PORT=SYMPH_PREFETCH_QUEUE_PORT,
            SYMPH_LOGGERPLEX_PORT=SYMPH_LOGGERPLEX_PORT,
            SYMPH_LOGGERPLEX_HOST=_LOCALHOST,
            SYMPH_TENSORPLEX_PORT=SYMPH_TENSORPLEX_PORT,
            SYMPH_TENSORPLEX_HOST=_LOCALHOST,
        ))

  def _get_replay_load_balancer(self):
    return ReplayLoadBalancer()

  def _get_exp_sender(self):
    return ExpSender(host=_LOCALHOST,
                     port=SYMPH_COLLECTOR_FRONTEND_PORT,
                     flush_iteration=1)

  def _get_learner_config(self):
    learner_config = ConfigDict()
    learner_config.replay = ConfigDict()
    learner_config.replay.memory_size = MAX_REPLAY_SIZE
    learner_config.replay.sampling_start_size = 0
    return learner_config

  def _get_session_config(self):
    session_config = ConfigDict()
    session_config.folder = '/tmp/replay_test'
    session_config.seed = 42

    session_config.replay = ConfigDict()
    session_config.replay.evict_interval = None
    session_config.replay.tensorboard_display = True

    session_config.loggerplex = ConfigDict()
    session_config.loggerplex.level = 'info'
    session_config.loggerplex.overwrite = True
    session_config.loggerplex.show_level = True
    session_config.loggerplex.time_format = 'hms'
    # enable_local_logger: print log to local stdout AND send to remote.
    session_config.loggerplex.enable_local_logger = True
    session_config.loggerplex.local_logger_level = session_config.loggerplex.level
    session_config.loggerplex.local_logger_time_format = session_config.loggerplex.time_format

    session_config.tensorplex = ConfigDict()
    session_config.tensorplex.max_processes = 2
    session_config.tensorplex.agent_bin_size = 4

    session_config.learner = ConfigDict()
    session_config.learner.max_prefetch_queue = 100
    session_config.learner.max_preprocess_queue = 100
    session_config.learner.prefetch_processes = 2

    return session_config

  def _get_uniform_replay(self):

    return UniformReplay(learner_config=self._get_learner_config(),
                         session_config=self._get_session_config(),
                         env_config=ConfigDict())

  def _get_data_fetcher(self):
    return LearnerDataPrefetcher(session_config=self._get_session_config(),
                                 batch_size=BATCH_SIZE,
                                 worker_preprocess=None,
                                 main_preprocess=lambda k: k)

  def _start_tensorplex_server(self):
    th = Thread(target=run_tensorplex, args=(self._get_session_config(), ))
    th.start()

  def _start_loggerplex_server(self):
    th = Thread(target=run_loggerplex, args=(self._get_session_config(), ))
    th.start()

  def testReplay(self):
    self._setup_env()
    self._start_loggerplex_server()
    self._start_tensorplex_server()

    rlb = self._get_replay_load_balancer()
    rlb.launch()

    exp_sender = self._get_exp_sender()

    replay = self._get_uniform_replay()
    replay.start_threads()

    df = self._get_data_fetcher()
    df.start()

    time.sleep(1)

    data = {'iteration': np.array([1.0])}
    for i in range(100):
      for j in range(BATCH_SIZE):
        exp_sender.send(data, {})

      time.sleep(.01)
      self.assertEqual(len(replay), min((1 + i) * BATCH_SIZE, MAX_REPLAY_SIZE))

      recv_data = df.get()
      self.assertEqual([data] * BATCH_SIZE, recv_data)
      print('Iteration %d successful' % i)

    print('Done!')


if __name__ == '__main__':
  tf.test.main()
