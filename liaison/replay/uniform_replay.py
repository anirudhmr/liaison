import random
from collections import deque

import liaison.utils as U
from absl import logging
from liaison.replay.base import Replay


class UniformReplay(Replay):

  def __init__(self, learner_config, env_config, session_config, index=0):
    """
        Args:
          memory_size: Max number of experience to store in the buffer.
            When the buffer overflows the old memories are dropped.
          sampling_start_size: min number of exp above which we will start sampling
        """
    super().__init__(learner_config=learner_config,
                     env_config=env_config,
                     session_config=session_config,
                     index=index)
    self.memory_size = self.learner_config.replay.memory_size
    self._memory = deque(maxlen=self.memory_size)
    self._next_idx = 0
    self.set_seed(session_config.seed)

  def insert(self, exp_dict):
    # appends to the right end of the queue
    self._memory.append(exp_dict)

  def sample(self, batch_size):
    indices = [
        random.randint(0,
                       len(self._memory) - 1) for _ in range(batch_size)
    ]
    response = [self._memory[i] for i in indices]
    return response

  def evict(self):
    raise NotImplementedError('no support for eviction in uniform replay mode')

  def start_sample_condition(self):
    return len(self) > self.learner_config.replay.sampling_start_size

  def __len__(self):
    return len(self._memory)

  def set_seed(self, seed):
    random.seed(seed)
