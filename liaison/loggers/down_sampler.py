import logging
from collections import deque

import liaison.utils as U
import numpy as np
from liaison.session.tracker import PeriodicTracker
from tensorflow.contrib.framework import nest

from liaison.loggers import BaseLogger


class Logger(BaseLogger):

  def __init__(self, base, period, is_average=False, window_size=None):
    """
      Sends write calls to the base logger or base loggers if
      base is list only once every period.

      Args:
        base: Logger or list of loggers
        period: int specifying period in counts of # of write calls.
        is_average: avg the collected history of values before
                    sending to base loggers.
        window_size: (Optional) Avg over this window.
                     If None, use entire period as default.
    """
    super(Logger, self).__init__()
    if isinstance(base, list):
      self._base_loggers = base
    else:
      self._base_loggers = [base]

    self._period = period
    self._is_average = is_average
    self._tracker = PeriodicTracker(period)
    self._history = None
    if is_average:
      maxlen = period
      if window_size is not None:
        maxlen = window_size
      self._history_spec = lambda k: deque(k, maxlen=maxlen)

  def _make_history(self, val):
    return self._history_spec(val)

  def _add_to_history(self, hist, val):
    return hist.append(val)

  def write(self, values, step=None):
    if step is None:
      step = self._step

    if self._tracker.track_increment():
      # Note that this is #write calls % period
      # and not necessarily step % period

      # compute the to_write value
      if self._is_average:
        if self._history is None:
          self._history = nest.map_structure(self._make_history, values)
        else:
          self._history = nest.map_structure(self._add_to_history,
                                             self._history, values)

        to_write = nest.map_structure(np.mean, self._history)
      else:
        to_write = values

      # write to base now
      for logger in self._base_loggers:
        logger.write(to_write, step=step)

    self._step += 1
