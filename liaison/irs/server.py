"""Start IRS Server."""

from __future__ import absolute_import, division, print_function

import os
import sys
from multiprocessing import Process

import liaison.utils as U
from absl import logging
from caraml.zmq import ZmqProxyThread
from git import Repo
from liaison.irs import IRSWorker
from liaison.utils import ConfigDict


"""
  Request format:
    (request_type -> str, args -> List, kwargs -> Dict)

"""


class Server(object):

  def __init__(self, results_folder, agent_config, env_config, sess_config,
               exp_name, exp_id, work_id, n_shards, **kwargs):
    """Results folder should be for the current work unit."""
    self.config = ConfigDict(**kwargs)
    self.n_shards = n_shards

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
                           results_folder=results_folder,
                           experiment_name=exp_name,
                           exp_id=exp_id,
                           work_id=work_id,
                           irs_n_shards=n_shards)
    self._register_src()
    self._register_cmd()

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
    for i in range(self.n_shards):
      worker = IRSWorker(serving_host='localhost',
                         serving_port=self.backend_port,
                         **self.config)
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

  def _register_configs(self, agent_config, env_config, sess_config, **kwargs):
    config_folder = self.config.configs_folder
    U.f_mkdir(config_folder)
    U.pretty_dump(agent_config, os.path.join(config_folder,
                                             'agent_config.json'))

    U.pretty_dump(env_config, os.path.join(config_folder, 'env_config.json'))

    U.pretty_dump(sess_config, os.path.join(config_folder, 'sess_config.json'))

    U.pretty_dump(kwargs, os.path.join(config_folder, 'misc_config.json'))

  def _register_cmd(self):
    cmd_folder = self.config.cmd_folder
    U.f_mkdir(cmd_folder)
    with open(os.path.join(cmd_folder, 'cmd.txt'), 'w') as f:
      f.write('\n'.join(sys.argv))

  def _register_src(self):
    src_folder = self.config.src_folder
    U.f_mkdir(src_folder)
    repo = Repo('./')
    commit = repo.head.commit
    src = dict(
        branch_name=repo.active_branch.name,
        commit_summary=commit.summary,
        commit_id=str(commit),
        commit_datetime=commit.committed_datetime.strftime(
            '%Y-%m-%d %H:%M:%S UTC'),
    )
    U.pretty_dump(src, os.path.join(src_folder, 'git_info.txt'))

    with open(os.path.join(src_folder, 'git_diff.txt'), 'w') as f:
      f.write(repo.git.diff(repo.head.commit.tree))

    os.system('conda list > %s' %
              os.path.join(src_folder, 'conda_env_list.txt'))
    U.compress_tar('./liaison/', os.path.join(src_folder, 'liaison.tar.gz'))
