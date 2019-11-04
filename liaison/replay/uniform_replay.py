import threading
import random
from collections import deque

import liaison.utils as U
from absl import logging
from liaison.replay.base import Replay, ReplayUnderFlowException


class UniformReplay(Replay):

  def __init__(self, seed, **kwargs):
    super().__init__(seed=seed, **kwargs)
    self._memory = deque(maxlen=self.config.memory_size)
    self.lock = threading.Lock()
    self._next_idx = 0
    self.set_seed(seed)
    if self._evict_interval:
      raise Exception("evict interval should be None for uniform replay.")

  def insert(self, exp_dict):
    # appends to the right end of the queue
    with self.lock:
      self._memory.append(exp_dict)

  def sample(self, batch_size):
    with self.lock:
      if len(self._memory) < batch_size:
        print('replay under flow exception encountered!')
        raise ReplayUnderFlowException()

      indices = [
          random.randint(0,
                         len(self._memory) - 1) for _ in range(batch_size)
      ]
      response = [self._memory[i] for i in indices]
      return response

  def evict(self):
    raise NotImplementedError('no support for eviction in uniform replay mode')

  def start_sample_condition(self):
    return len(self) > self.config.sampling_start_size

  def __len__(self):
    return len(self._memory)

  def set_seed(self, seed):
    random.seed(seed)
