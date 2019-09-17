import logging

import liaison.utils as U

from .base import BaseLogger


class Logger(BaseLogger):

  def __init__(self, name=None):
    super(Logger, self).__init__()
    self._name = name
    if name is not None:
      assert isinstance(name, str)

  def write(self, values, step=None):
    if step is None:
      step = self._step
    logging.info(
        'Values %sat step %d: {}'.format(values),
        'for ' + self._name + ' ' if self._name else '',
        step,
    )
    self._step += 1
