import logging
import os

from caraml.zmq import ZmqClient, ZmqTimeoutError


class Client:

  def __init__(self,
               host=os.environ['SYMPH_IRS_FRONTEND_HOST'],
               port=os.environ['SYMPH_IRS_FRONTEND_PORT']):
    self._cli = ZmqClient(host=host,
                          port=port,
                          serializer='pyarrow',
                          deserializer='pyarrow',
                          timeout=2)

  def _send_to_server(self, req, *args, **kwargs):
    while True:
      try:
        return self._cli.request([req, args, kwargs])
      except ZmqTimeoutError:
        logging.info("Timeout error encountered in IRS Client .. Retrying")

  def __getattr__(self, attr):
    """Redirect requests to the remote."""
    # TODO: add doc checking
    return lambda *args, **kwargs: self._send_to_server(attr, *args, **kwargs)
