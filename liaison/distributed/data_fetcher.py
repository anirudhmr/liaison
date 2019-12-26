import os
import queue
from threading import Thread

import liaison.utils as U
from caraml.zmq import DataFetcher

from .exp_serializer import get_deserializer, get_serializer


class LearnerDataPrefetcher(DataFetcher):
  """
        Convenience class that initializes everything from session config
        + batch_size

        Fetches data from replay in multiple processes and put them into
        a queue

        First spawns worker_preprocess
    """

  def __init__(
      self,
      batch_size,
      prefetch_batch_size,
      combine_trajs,
      max_prefetch_queue,
      prefetch_processes,
      prefetch_threads_per_process,
      tmp_dir,
      worker_preprocess=None,
  ):
    assert batch_size % prefetch_batch_size == 0
    self.fetch_queue = queue.Queue(
        maxsize=max(1, max_prefetch_queue - batch_size // prefetch_batch_size))
    self._combine_prefetch_queue = queue.Queue(maxsize=16)
    self.timer = U.TimeRecorder()

    self.sampler_host = os.environ['SYMPH_SAMPLER_FRONTEND_HOST']
    self.sampler_port = os.environ['SYMPH_SAMPLER_FRONTEND_PORT']
    self._combine_trajs = combine_trajs
    self.batch_size = batch_size
    self.prefetch_batch_size = prefetch_batch_size
    self.prefetch_processes = prefetch_processes
    self.prefetch_host = '127.0.0.1'
    self.worker_comm_port = os.environ['SYMPH_PREFETCH_QUEUE_PORT']
    self.worker_preprocess = worker_preprocess
    super().__init__(handler=self._put,
                     remote_host=self.sampler_host,
                     remote_port=self.sampler_port,
                     requests=self.request_generator(),
                     worker_comm_port=self.worker_comm_port,
                     remote_serializer=get_serializer(),
                     remote_deserialzer=get_deserializer(),
                     n_workers=self.prefetch_processes,
                     worker_handler=self.worker_preprocess,
                     threads_per_worker=prefetch_threads_per_process,
                     tmp_dir=tmp_dir)

  def run(self):
    self._combine_prefetch_thread = Thread(
        target=self._combine_prefetched_batches)
    self._combine_prefetch_thread.start()
    super().run()

  def _put(self, _, data):
    self.fetch_queue.put(data, block=True)

  def _combine_prefetched_batches(self):
    while True:
      l = []
      while len(l) < self.batch_size:
        l.extend(self.fetch_queue.get().data)
      self._combine_prefetch_queue.put(self._combine_trajs(l))

  def get(self):
    with self.timer.time():
      return self._combine_prefetch_queue.get()

  def request_generator(self):
    while True:
      yield self.prefetch_batch_size
