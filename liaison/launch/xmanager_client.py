from xmanager import Server
from caraml.zmq import get_remote_client


def get_xmanager_client(*args, **kwargs):
  return get_remote_client(Server, *args, **kwargs)
