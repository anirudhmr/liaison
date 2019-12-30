from caraml.zmq import get_remote_client

try:
  from xmanager.xmanager.server import XManagerServer
except ImportError:
  from xmanager.server import XManagerServer


def get_xmanager_client(*args, **kwargs):
  return get_remote_client(XManagerServer, *args, **kwargs)
