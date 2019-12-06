import os
import time

import numpy as np

import liaison.utils as U
import tree as nest
from liaison.loggers import BaseLogger
from tensorplex import LoggerplexClient, TensorplexClient


class Logger(BaseLogger):

  def __init__(self, stream_id, client):
    """
      sends each write as a new stream to client.record_kv_data
    """
    super(Logger, self).__init__()
    self._stream_id = stream_id
    self._client = client

  def write(self, dict_values, step=None):
    if step is None:
      step = self._step

    self._client.record_kv_data(self._stream_id + f'-{step}',
                                dict_values,
                                time=time.time())
    self._step += 1
