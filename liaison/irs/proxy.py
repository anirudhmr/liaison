import json
import logging
import os
import pickle
import shutil
import time
from pathlib import Path
from threading import Thread

import liaison.utils as U
from absl import logging
from caraml.zmq import ZmqProxyThread, ZmqServer
from liaison.irs.client import Client as IRSClient
from liaison.utils import ConfigDict


class Proxy:

  def __init__(self, serving_host, serving_port):
    self.serving_host = serving_host
    self.serving_port = serving_port
    self._irs_client = IRSClient(auto_detect_proxy=False)

  def run(self):
    self._server = ZmqServer(host='*',
                             port=self.serving_port,
                             serializer=U.serialize,
                             deserializer=U.deserialize,
                             bind=True)
    self._server.start_loop(handler=self._handle_request, blocking=True)

  def _handle_request(self, req):
    req_fn, args, kwargs = req
    return getattr(self._irs_client, req_fn)(*args, **kwargs)
