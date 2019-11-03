"""Batch several copies of environment together to run in parallel."""

from __future__ import absolute_import, division, print_function

import multiprocessing as mp
import threading
from multiprocessing import Queue

from liaison.env.batch import BaseBatchedEnv


class EnvWorker:

  def __init__(self, send_queue, recv_queue, id, seed, env_class, env_config):
    self._send_queue = send_queue
    self._recv_queue = recv_queue
    self._id = id
    self._env = env_class(id=self._id, seed=seed, **env_config)
    self._start()

  def _start(self):
    while True:
      func_name, args, kwargs = self._recv_queue.get()
      self._send_queue.put([getattr(self._env, func_name)(*args, **kwargs)])


class BatchedEnv(BaseBatchedEnv):

  def __init__(self,
               n_envs,
               env_class,
               env_configs,
               seed,
               use_threads=False,
               **kwargs):

    super(BatchedEnv, self).__init__(n_envs, env_class, env_configs, seed)

    if use_threads:
      Runnable = threading.Thread
    else:
      Runnable = mp.Process

    self._n_workers = n_envs
    self._workers = []
    self._send_queues = []
    self._recv_queues = []
    for i in range(self._n_workers):
      send_queue = Queue(1)
      recv_queue = Queue(1)
      worker = Runnable(target=EnvWorker,
                        args=(recv_queue, send_queue, i, seed, env_class,
                              env_configs[i]))
      worker.start()
      self._workers.append(worker)
      self._send_queues.append(send_queue)
      self._recv_queues.append(recv_queue)

    self._setup_obs_spec()
    self._setup_action_spec()

    self._make_step_spec(self._obs_spec)
    self.set_seed(seed)

  def _send_to_workers(self, method, *args, **kwargs):

    n_workers = self._n_workers
    if len(args) == 0:
      args = [[]] * n_workers

    if len(kwargs) == 0:
      kwargs = [{}] * n_workers

    assert len(args) == n_workers
    assert len(kwargs) == n_workers

    for i, s_q in enumerate(self._send_queues):
      s_q.put([method, args[i], kwargs[i]])

    results = []
    for r_q in self._recv_queues:
      results.append(r_q.get()[0])
    return results

  def _setup_obs_spec(self):
    self._obs_spec = self._stack_specs(
        self._send_to_workers('observation_spec'))

  def _setup_action_spec(self):
    self._action_spec = self._stack_specs(self._send_to_workers('action_spec'))

  def step(self, action):
    return self._stack_ts(
        self._send_to_workers('step', *[(act, ) for act in action]))

  def reset(self):
    return self._stack_ts(self._send_to_workers('reset'))

  def set_seed(self, seed):
    return self._send_to_workers('set_seed', *[(seed, )] * self._n_workers)
