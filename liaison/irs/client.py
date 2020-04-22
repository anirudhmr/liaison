import logging
import os

from caraml.zmq import ZmqClient, ZmqTimeoutError


def get_irs_client(timeout, auto_detect_proxy=True):

  if 'SYMPH_IRS_PROXY_HOST' in os.environ and auto_detect_proxy:
    host = os.environ['SYMPH_IRS_PROXY_HOST']
    port = os.environ['SYMPH_IRS_PROXY_PORT']
  else:
    host = os.environ['SYMPH_IRS_HOST']
    port = os.environ['SYMPH_IRS_PORT']

  return ZmqClient(host=host,
                   port=port,
                   serializer='pyarrow',
                   deserializer='pyarrow',
                   timeout=timeout)


class Client:

  def __init__(self, timeout=2, auto_detect_proxy=True):
    self._cli = get_irs_client(timeout, auto_detect_proxy)

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
