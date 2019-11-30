"""TODO(arc): doc_string."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import sys
import utils as U
import numpy as np

from caraml.zmq import (ZmqClient, ZmqProxyThread, ZmqPub, ZmqServer, ZmqSub,
                        ZmqTimeoutError)
from absl import logging
from absl import app
from distributed import ParameterClient

PORT = 6001

var_dict = dict(x=np.array(0, dtype=np.int32), y=np.array(0, dtype=np.int32))


def get_ps_client():
  return ParameterClient(port=PORT, host='localhost', timeout=2)


def server_f(req):
  logging.info('Received req: {}'.format(req))
  args = []
  if isinstance(req, list):
    args = req[1]

  params_asked_for = {var_name: var_dict[var_name] for var_name in args}
  return params_asked_for, {
      'time': time.time(),
      'hash': U.pyobj_hash(var_dict)
  }


def main(_):
  server = ZmqServer(host='localhost',
                     port=PORT,
                     serializer=U.serialize,
                     deserializer=U.deserialize,
                     bind=True)
  server_thread = server.start_loop(handler=server_f, blocking=False)

  # client = ZmqClient(host='localhost',
  #                    port=PORT,
  #                    timeout=2,
  #                    serializer=U.serialize,
  #                    deserializer=U.deserialize)
  client = get_ps_client()
  for _ in range(10):
    # client.request(['info', ['x']])
    client.fetch_parameter_with_info(['x'])
  print('Done!')


if __name__ == '__main__':
  app.run(main)
