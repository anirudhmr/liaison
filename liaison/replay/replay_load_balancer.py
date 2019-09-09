import os
from multiprocessing import Process

from caraml.zmq import ZmqProxyThread


class ReplayLoadBalancer(object):

  def __init__(self):
    self.sampler_proxy = None
    self.collector_proxy = None

    self.collector_frontend_port = os.environ['SYMPH_COLLECTOR_FRONTEND_PORT']
    self.collector_backend_port = os.environ['SYMPH_COLLECTOR_BACKEND_PORT']
    self.sampler_frontend_port = os.environ['SYMPH_SAMPLER_FRONTEND_PORT']
    self.sampler_backend_port = os.environ['SYMPH_SAMPLER_BACKEND_PORT']

    self.collector_frontend_add = "tcp://*:{}".format(
        self.collector_frontend_port)
    self.collector_backend_add = "tcp://*:{}".format(
        self.collector_backend_port)
    self.sampler_frontend_add = "tcp://*:{}".format(self.sampler_frontend_port)
    self.sampler_backend_add = "tcp://*:{}".format(self.sampler_backend_port)

  def launch(self):
    self.collector_proxy = ZmqProxyThread(in_add=self.collector_frontend_add,
                                          out_add=self.collector_backend_add,
                                          pattern='router-dealer')
    self.sampler_proxy = ZmqProxyThread(in_add=self.sampler_frontend_add,
                                        out_add=self.sampler_backend_add,
                                        pattern='router-dealer')

    self.collector_proxy.setDaemon(False)
    self.collector_proxy.start()
    self.sampler_proxy.setDaemon(False)
    self.sampler_proxy.start()

  def join(self):
    self.collector_proxy.join()
    self.sampler_proxy.join()
