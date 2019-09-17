"""Start IRS Server."""

from __future__ import absolute_import, division, print_function

import os
from multiprocessing import Process

from liaison.irs import IRSWorker
import liaison.utils as U
from absl import logging
from caraml.zmq import ZmqProxyThread
from liaison.utils import ConfigDict
"""
  Request format:
    (request_type -> str, args -> List, kwargs -> Dict)

"""


class Server(object):

  def __init__(self, results_folder, agent_config, env_config, sess_config,
               network_config, exp_name, exp_id, work_id, shards, **kwargs):
    """Results folder should be for the current work unit."""
    self.config = ConfigDict(**kwargs)
    self.results_folder = results_folder
    self.shards = shards

    # Serving parameter to agents
    self.frontend_port = os.environ['SYMPH_IRS_FRONTEND_PORT']
    self.backend_port = os.environ['SYMPH_IRS_BACKEND_PORT']
    self.serving_frontend_add = "tcp://*:{}".format(self.frontend_port)
    self.serving_backend_add = "tcp://*:{}".format(self.backend_port)

    self.proxy = None
    self.workers = []
    self._register_configs(agent_config,
                           env_config,
                           sess_config,
                           network_config,
                           results_folder=results_folder,
                           experiment_name=exp_name,
                           exp_id=exp_id,
                           work_id=work_id,
                           irs_n_shards=shards,
                           **kwargs)

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
      worker = IRSWorker(serving_host='localhost',
                         serving_port=self.backend_port,
                         results_folder=self.results_folder)
      worker.start()
      self.workers.append(worker)

  def join(self):
    """
        Wait for all workers to exit
            (Currently this means they crashed)
        Note that proxy is a daemon thread and doesn't need waiting
      """
    for i, worker in enumerate(self.workers):
      worker.join()
      U.report_exitcode(worker.exitcode, 'ps-{}'.format(i))

  def quit(self):
    for worker in self.workers:
      worker.terminate()

  def _register_configs(self, agent_config, env_config, sess_config,
                        network_config, **kwargs):
    U.f_mkdir(os.path.join(self.results_folder, 'configs'))
    U.pretty_dump(
        agent_config,
        os.path.join(self.results_folder, 'configs', 'agent_config.json'))

    U.pretty_dump(
        env_config,
        os.path.join(self.results_folder, 'configs', 'env_config.json'))

    U.pretty_dump(
        sess_config,
        os.path.join(self.results_folder, 'configs', 'sess_config.json'))

    U.pretty_dump(
        network_config,
        os.path.join(self.results_folder, 'configs', 'network_config.json'))

    U.pretty_dump(
        kwargs, os.path.join(self.results_folder, 'configs',
                             'misc_config.json'))

  def _register_src_code(self):
    pass
