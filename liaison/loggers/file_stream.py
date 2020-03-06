import os
import time

import numpy as np

import liaison.utils as U
import tree as nest
from liaison.loggers import BaseLogger
from tensorplex import LoggerplexClient, TensorplexClient


class Logger(BaseLogger):

  def __init__(self, client):
    """
      sends each write as a new stream to client.save_file
    """
    super(Logger, self).__init__()
    self._client = client

  def write(self, f, fname, step=None):
    if step is None:
      step = self._step

    self._client.save_file(fname, f.read(-1))

    self._step += 1
