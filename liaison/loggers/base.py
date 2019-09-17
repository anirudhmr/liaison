class Logger:

  def __init__(self):
    self._step = 0

  def write(self, values, step=None):
    """
      Write a dict of key-value pairs at global_step.
      If `global_step` is not provided, then use internal
      clock to set it.
    """
    raise NotImplementedError
