import os
import liaison.utils as U
from tensorplex import LoggerplexClient, TensorplexClient

from liaison.loggers import BaseLogger


class Logger(BaseLogger):

  def __init__(self,
               client_id,
               host=os.environ['SYMPH_TENSORPLEX_HOST'],
               port=os.environ['SYMPH_TENSORPLEX_PORT']):
    super(Logger, self).__init__()
    self._client_id = client_id
    self._client = TensorplexClient(client_id, host, port)

  def write(self, values, step=None):
    if step is None:
      step = self._step
    self._client.add_scalars(values, global_step=step)
    self._step += 1
