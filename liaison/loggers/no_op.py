from liaison.loggers import BaseLogger


class Logger(BaseLogger):

  def __init__(self):
    super(BaseLogger, self).__init__()

  def write(self, vals, step=None):
    return
