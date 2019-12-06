import numpy as np

import tree as nest
from liaison.loggers import BaseLogger


class AvgLogger(BaseLogger):

  def __init__(self, logger: BaseLogger):
    """Pipes inputs to the logger after modification."""
    self._logger = logger

  def write(self, dict_values, step=None):
    dict_values = nest.map_structure(np.mean, dict_values)
    return self._logger.write(dict_values, step)
