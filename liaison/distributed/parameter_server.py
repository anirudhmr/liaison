"""
    Defines the parameter publishing mechanism that propagates
        updated parameters from the learner to agents
"""
import os
import sys
import time
from multiprocessing import Process

import utils as U
from absl import logging
from caraml.zmq import (ZmqClient, ZmqProxyThread, ZmqPub, ZmqServer, ZmqSub,
                        ZmqTimeoutError)


class ParameterPublisher(object):
  """
        Publishes parameters from the learner side
        Using ZmqPub socket
    """

  def __init__(self, port):
    """
        Args:
            port: the port connected to the pub socket
            module_dict: ModuleDict object that exposes model parameters
        """
    self._publisher = ZmqPub(
        host='*',
        port=port,
        serializer=U.serialize,
    )

  def publish(self, iteration, var_dict, message=''):
    """
        Called by learner. Publishes model parameters with additional info

        Args:
            iteration: current learning iteration
            var_dict: Dict of available variables.
            message: any U.serialize serializable data
        """
    info = {
        'time': time.time(),
        'iteration': iteration,
        'message': message,
        'variable_list': list(var_dict.keys()),
        'hash': U.pyobj_hash(var_dict),
    }
    self._publisher.pub(topic='ps', data=(var_dict, info))


class ShardedParameterServer(object):
  """
        Runs multiple parameter servers in parallel processes.
    """

  def __init__(self, shards, supress_output=False):
    self.shards = shards

    # Serving parameter to agents
    self.frontend_port = os.environ['SYMPH_PS_FRONTEND_PORT']
    self.backend_port = os.environ['SYMPH_PS_BACKEND_PORT']
    self.serving_frontend_add = "tcp://*:{}".format(self.frontend_port)
    self.serving_backend_add = "tcp://*:{}".format(self.backend_port)

    # Subscribing to learner published parameters
    self.publisher_host = os.environ['SYMPH_PARAMETER_PUBLISH_HOST']
    self.publisher_port = os.environ['SYMPH_PARAMETER_PUBLISH_PORT']

    self._supress_output = supress_output
    self.proxy = None
    self.workers = []

  def launch(self):
    """
            Runs load balancing proxy thread
                and self.shards ParameterServer processes
            Returns after all threads and processes are running
        """
    self.proxy = ZmqProxyThread(in_add=self.serving_frontend_add,
                                out_add=self.serving_backend_add,
                                pattern='router-dealer')
    self.proxy.start()

    self.workers = []
    for i in range(self.shards):
      worker = ParameterServer(publisher_host=self.publisher_host,
                               publisher_port=self.publisher_port,
                               serving_host='localhost',
                               serving_port=self.backend_port,
                               load_balanced=True,
                               supress_output=self._supress_output)
      worker.start()
      self.workers.append(worker)

  def join(self):
    """
            Wait for all parameter server workers to exit
                (Currently this means they crashed)
            Note that proxy is a daemon thread and doesn't need waiting
        """
    for i, worker in enumerate(self.workers):
      worker.join()
      U.report_exitcode(worker.exitcode, 'ps-{}'.format(i))

  def quit(self):
    for worker in self.workers:
      worker.terminate()


class ParameterServer(Process):
  """
        Standalone script for PS node that runs in an infinite loop.
        The ParameterServer subscribes to learner to get the latest
            model parameters and serves these parameters to agents
        It implements a simple hash based caching mechanism to avoid
            serving duplicate parameters to agent
    """

  def __init__(
      self,
      publisher_host,
      publisher_port,
      serving_host,
      serving_port,
      load_balanced=False,
      supress_output=False,
  ):
    """
        Args:
            publisher_host, publisher_port: where learner publish parameters
            serving_host, serving_port: where to serve parameters to agents
            load_balanced: whether multiple parameter servers are sharing the
                same address
        """
    Process.__init__(self)
    self.publisher_host = publisher_host
    self.publisher_port = publisher_port
    self.serving_host = serving_host
    self.serving_port = serving_port
    self.load_balanced = load_balanced
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

    self._subscriber = ZmqSub(
        host=self.publisher_host,
        port=self.publisher_port,
        # handler=self._set_storage,
        topic='ps',
        deserializer=U.deserialize,
    )
    self._server = ZmqServer(
        host=self.serving_host,
        port=self.serving_port,
        # handler=self._handle_agent_request,
        serializer=U.serialize,
        deserializer=U.deserialize,
        bind=not self.load_balanced,
    )
    self._subscriber_thread = self._subscriber.start_loop(
        handler=self._set_storage, blocking=False)
    self._server_thread = self._server.start_loop(
        handler=self._handle_agent_request, blocking=False)
    logging.info('Parameter server started')

    self._subscriber_thread.join()
    self._server_thread.join()

  def _set_storage(self, data):
    logging.info('Set storage called on ps')
    self.parameters, self.param_info = data
    if self.param_info is None:
      logging.info('_set_storage received None info')
    else:
      logging.info('_set_storage received info: {}'.format(self.param_info))

  def _handle_agent_request(self, request):
    """
            Reply to agents' request for parameters

        Args:
            request: 3 types
             - "info": (None, info)
             - "parameter": (param, info)
             - "parameter:<agent-hash>":
                returns (None, None) if no parameter has been published
                returns (None, info) if the hash
                    of server side parameters is the same as the agent's
                otherwise returns (param, info)
        """
    logging.info('Request received of type: %s', request)
    args = []
    if isinstance(request, list):
      args = request[1]
      request = request[0]

    if request == 'info':
      return None, self.param_info
    elif request.startswith('parameter'):
      if self.parameters is None:
        return None, None
      else:
        params_asked_for = {
            var_name: self.parameters[var_name]
            for var_name in args
        }
      if ':' in request:
        _, last_hash = request.split(':', 1)
        current_hash = self.param_info['hash']
        if last_hash == current_hash:  # param not changed
          return None, self.param_info
        else:
          return params_asked_for, self.param_info
      else:
        return params_asked_for, self.param_info
    else:
      raise ValueError('invalid request: ' + str(request))


class ParameterClient(object):
  """
        On agent side, sends requests to parameter servers to fetch the
        latest parameters.
    """

  def __init__(self, host, port, timeout=2):
    """
        Args:
            host: parameter server host
            port: parameter server port
            timeout: how long should the the client wait
                if the parameter server is not available
        """
    self.host = host
    self.port = port
    self.timeout = timeout
    self._current_info = {}
    self._last_hash = ''
    self.alive = False

    self._client = ZmqClient(host=self.host,
                             port=self.port,
                             timeout=self.timeout,
                             serializer=U.serialize,
                             deserializer=U.deserialize)

  def fetch_parameter_with_info(self, var_names, force_update=False):
    """
            Called by agent to retrieve parameters
            By default, pulls from PS ONLY WHEN the parameter hash changes
                to prevent duplicate fetching. No-op when duplicate.
            Caching can be overriden by force_update

        Args:
            force_update: forces download of parameter, regardless of
                currently cached hash

        Returns:
            (param or None, info or None)
        """
    try:
      if force_update:
        response = self._client.request(['parameter', var_names])
      else:
        response = self._client.request(
            ['parameter:' + self._last_hash, var_names])
    except ZmqTimeoutError:
      logging.info('ZmQ timed out.')
      self.on_fetch_parameter_failed()
      return None, None
    self.on_fetch_parameter_success()
    param, info = response
    if info is None:
      return None, None

    self._last_hash = info['hash']
    return param, info

  def fetch_info(self):
    """
            Fetch the metadata of parameters on parameter server

        Returns:
            dictionary of metadata
        """
    try:
      response = self._client.request('info')
    except ZmqTimeoutError:
      logging.info('ZmQ timed out.')
      self.on_fetch_parameter_failed()
      return None
    self.on_fetch_parameter_success()
    _, info = response
    return info

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
