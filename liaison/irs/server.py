"""Start IRS Server."""

import os
import sys
from multiprocessing import Process
from pathlib import Path

import liaison.utils as U
from absl import logging
from liaison.irs import IRSWorker
from liaison.launch.xmanager_client import get_xmanager_client
from liaison.utils import ConfigDict


"""
  Request format:
    (request_type -> str, args -> List, kwargs -> Dict)

"""


class Server:

  def __init__(self, results_folder, agent_config, env_config, sess_config,
               exp_name, exp_id, work_id, n_shards, hyper_params, **kwargs):
    """Results folder should be for the current work unit."""
    self.config = ConfigDict(**kwargs)
    self.n_shards = n_shards

    # Serving parameter to agents
    self.port = os.environ['SYMPH_IRS_PORT']
    self._register_configs(agent_config,
                           env_config,
                           sess_config,
                           results_folder=results_folder,
                           experiment_name=exp_name,
                           exp_id=exp_id,
                           work_id=work_id,
                           irs_n_shards=n_shards)
    self._register_cmd()
    if work_id == 0:
      self._register_src()
      U.f_mkdir(results_folder)
      with open(f'{results_folder}/hyper_params.json', 'w') as f:
        f.write(hyper_params)

  def launch(self):
    """
        Runs load balancing proxy thread
            and self.shards ParameterServer processes
        Returns after all threads and processes are running
    """
    self.worker = IRSWorker(serving_host='*',
                            serving_port=self.port,
                            **self.config)
    self.worker.start()

  def join(self):
    """
      Wait for worker to exit
          (Currently this means they crashed)
      Note that proxy is a daemon thread and doesn't need waiting
    """
    worker = self.worker
    worker.join()
    U.report_exitcode(worker.exitcode, 'ps-{}'.format(i))

  def quit(self):
    self.worker.terminate()

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
    # this import requires fetching git executable which might not be available
    # on all systems.
    from git import Repo
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

  def _register_xmanager_record(self, exp_id):
    cli = get_xmanager_client(host=os.environ['XMANAGER_HOST'],
                              port=int(os.environ['XMANAGER_PORT']),
                              serializer='pyarrow',
                              deserializer='pyarrow')
    rec = cli.fetch_record(int(exp_id))
    path = Path(self.config.xmanager_record_path)
    path.parent.mkdir(parents=False, exist_ok=False)
    U.pretty_dump(rec, path)
    print('xmanager record received')
