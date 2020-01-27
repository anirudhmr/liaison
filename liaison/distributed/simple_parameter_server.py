"""
Simplify liaison.distributed.parameter_server by removing sharding
, proxy and load balancing.
"""

import os
import sys
import time
from collections import namedtuple
from threading import Thread

import liaison.utils as U
from absl import logging
from caraml.zmq import ZmqClient, ZmqProxyThread, ZmqServer, ZmqTimeoutError
from liaison.distributed.parameter_server import PSRequest, PSResponse

# PublishRequest = namedtuple(
#     'PublishRequest',
#     ['time', 'iteration', 'variable_list', 'agent_scope', 'hash'])


class ParameterPublisher(object):
  """
      Publishes parameters from the learner side
      Using ZmqPub socket
  """

  def __init__(self, host, port, agent_scope):
    """
        Args:
            host: IP of the ps
            port: the port connected to the pub socket
        """
    self._agent_scope = agent_scope
    self.alive = False

    self._publisher = ZmqClient(
        host=host,
        port=port,
        timeout=2,
        serializer=U.serialize,
        deserializer=U.deserialize,
    )

  def publish(self, iteration, var_dict):
    """
        Called by learner. Publishes model parameters with additional info

        Args:
            iteration: current learning iteration
            var_dict: Dict of available variables.
    """
    info = {
        'agent_scope': self._agent_scope,
        'time': time.time(),
        'iteration': iteration,
        'variable_list': list(var_dict.keys()),
        'hash': U.pyobj_hash(var_dict),
    }
    while True:
      try:
        self._publisher.request((var_dict, info))
      except ZmqTimeoutError as e:
        self.on_fetch_parameter_failed()
        continue
      break
    self.on_fetch_parameter_success()

  def on_fetch_parameter_failed(self):
    """
            Called when connection with parameter server fails
            to be established
        """
    if self.alive:
      self.alive = False
      logging.info('Parameter client request timed out')

  def on_fetch_parameter_success(self):
    """
            Called when connection with parameter server
            is succesfully established
        """
    if not self.alive:
      self.alive = True
      logging.info('Parameter client came back alive')


class ParameterServer(Thread):
  """
      Standalone script for PS node that runs in an infinite loop.
      The ParameterServer subscribes to learner to get the latest
          model parameters and serves these parameters to agents
      It implements a simple hash based caching mechanism to avoid
          serving duplicate parameters to agent
  """

  def __init__(
      self,
      publish_port,
      serving_port,
      supress_output=False,
  ):
    """
        Args:
            publish_port: where learner should send parameters to.
            load_balanced: whether multiple parameter servers are sharing the
                same address
        """
    Thread.__init__(self)
    self.publish_port = publish_port
    self.serving_port = serving_port
    self._supress_output = supress_output
    # storage
    self.parameters = None
    self.param_info = None
    # threads
    self._subscriber = None
    self._server = None
    self._subscriber_thread = None
    self._server_thread = None

  def run(self):
    """
      Run relative threads and wait until they finish (due to error)
    """

    if self._supress_output:
      sys.stdout = open('/tmp/' + 'latest' + ".out", "w")
      sys.stderr = open('/tmp/' + 'latest' + ".err", "w")

    self._param_reciever = ZmqServer(
        host='*',
        port=self.publish_port,
        serializer=U.serialize,
        deserializer=U.deserialize,
    )
    self._server = ZmqServer(
        host='*',
        port=self.serving_port,
        # handler=self._handle_agent_request,
        serializer=U.serialize,
        deserializer=U.deserialize,
    )
    self._subscriber_thread = self._param_reciever.start_loop(
        handler=self._set_storage, blocking=False)
    self._server_thread = self._server.start_loop(
        handler=self._handle_agent_request, blocking=False)
    logging.info('Parameter server started')

    self._subscriber_thread.join()
    self._server_thread.join()

  def _set_storage(self, data):
    self.parameters, self.param_info = data
    logging.info('_set_storage received info: {}'.format(self.param_info))

  def _handle_agent_request(self, request):
    """Reply to agents' request for parameters."""

    request = PSRequest(**request)
    logging.info('Request received of type: %s', request.type)

    if self.param_info is None:
      return PSResponse(type='not_ready')._asdict()

    if request.type == 'info':
      return PSResponse(type='info', info=self.param_info)._asdict()

    elif request.type == 'parameter':
      if request.hash is not None:
        if request.hash == self.param_info['hash']:  # param not changed
          return PSResponse(type='no_change', info=self.param_info)._asdict()

      params_asked_for = {
          var_name: self.parameters[var_name.replace(
              request.agent_scope + '/', self.param_info['agent_scope'] + '/',
              1)]
          for var_name in request.var_list
      }
      return PSResponse(type='parameters',
                        info=self.param_info,
                        parameters=params_asked_for)._asdict()

    else:
      raise ValueError('invalid request type received: %s' % (request.type))
