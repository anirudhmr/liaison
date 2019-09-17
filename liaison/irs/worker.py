"""Start IRS Server."""

from __future__ import absolute_import, division, print_function

import logging
import os
from multiprocessing import Process

import liaison.utils as U
from absl import logging
from caraml.zmq import ZmqProxyThread
from liaison.utils import ConfigDict
"""
  Request format:
    (request_type -> str, args -> List, kwargs -> Dict)

"""


class Worker(Process):

  def __init__(self, serving_host, serving_port):
    Process.__init__(self)
    self.serving_host = serving_host
    self.serving_port = serving_port

    # Attributes
    self._server = None

  def run(self):
    self._server = ZmqServer(host=self.serving_host,
                             port=self.serving_port,
                             serializer=U.serialize,
                             deserializer=U.deserialize,
                             bind=False)
    self._server.start_loop(handler=self._handle_request, blocking=True)

  def _handle_request(self, req):
    req_fn, args, kwargs = req
    assert isinstance(req_fn, str)
    try:
      fn = getattr(self, req_fn)
      return fn(*args, **kwargs)
    except AttributeError:
      logging.error('Unknown request func name received: %s', req_fn)

  # ================== PUBLIC REMOTE API ==================
