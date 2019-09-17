"""Entry script for each component in the setup."""

from __future__ import absolute_import, division, print_function

import liaison.utils as U
from liaison.utils import ConfigDict
from absl import logging, app
from argon import ArgumentParser, to_nested_dicts
from liaison.launch.launcher import Launcher

parser = ArgumentParser('main entry script for all spawned liaison processes.')
# agent_config
parser.add_config_file(name='agent', required=True)

# env_config
parser.add_config_file(name='env', required=True)

# sess_config
parser.add_config_file(name='sess', required=True)

# network_config
parser.add_config_file(name='network', required=True)

parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--traj_length',
                    type=int,
                    default=100,
                    help='Length of the experience trajectories')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--experiment_id', type=int, required=True)
parser.add_argument('--experiment_name', type=str, required=True)
parser.add_argument('--work_id', type=int, required=True)
parser.add_argument('--results_folder', type=str, required=True)
parser.add_argument('--n_actors', type=int)


class LauncherSetup(Launcher):

  def __init__(self):
    pass

  def setup(self, argv):
    """After reading the component name this function will be called."""

    args = parser.parse_args(args=argv)

    self.args = args
    self.experiment_id = args.experiment_id
    self.work_id = args.work_id
    self.experiment_name = args.experiment_name
    self.batch_size = args.batch_size
    self.traj_length = args.traj_length
    self.seed = args.seed
    self.results_folder = args.results_folder
    self.n_actors = args.n_actors

    self.agent_config = ConfigDict(to_nested_dicts(args.agent_config))
    self.env_config = ConfigDict(to_nested_dicts(args.env_config))
    self.sess_config = ConfigDict(to_nested_dicts(args.sess_config))
    self.network_config = ConfigDict(to_nested_dicts(args.network_config))


def main(_):
  LauncherSetup().main()


if __name__ == "__main__":
  app.run(main)
