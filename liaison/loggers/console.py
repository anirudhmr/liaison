import logging
import time

import liaison.utils as U

from . import BaseLogger


class Logger(BaseLogger):

  def __init__(self, print_every=None, name=None):
    """
      Args:
        print_every: Prints every this many seconds.
    """
    super(Logger, self).__init__()
    self._name = name
    self._print_every = print_every
    self._last_print_time = time.time()
    if name is not None:
      assert isinstance(name, str)

  def _log(self, values, step):
    logging.info('Values %sat step %d: {}'.format(values),
                 'for ' + self._name + ' ' if self._name else '', step)

  def write(self, values, step=None):
    if step is None:
      step = self._step

    if self._print_every is not None:
      if time.time() - self._last_print_time >= self._print_every:
        self._log(values, step)
        self._last_print_time = time.time()
    else:
      self._log(values, step)
    self._step += 1
