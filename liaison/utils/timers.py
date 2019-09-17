import time


class Timer(object):

  def __init__(self, verbose=False):
    self.verbose = verbose
    self.elapsed = 0.0

  def __enter__(self):
    self.elapsed = 0
    self.elapsed_secs = 0
    self.start = time.clock()
    return self

  def __exit__(self, *args):
    self.elapsed_secs = time.clock() - self.start
    self.elapsed = float(self.elapsed_secs * 1000)  # millisecs
    if self.verbose:
      print('elapsed time: %f ms' % self.elapsed)

  def to_seconds(self):
    return self.elapsed_secs

  def to_ms(self):
    return float(self.elapsed)
