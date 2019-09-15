import math
import os
import socket
import sys
from copy import copy

import liaison.utils as U
from argon import ArgumentParser, to_nested_dicts
from liaison.launch import CommandGenerator, setup_network
from liaison.launch.xmanager_client import XManagerClient
from liaison.utils import ConfigDict
from symphony.commandline import SymphonyParser
from symphony.engine import Cluster

ENV_ACTIVATE_CMD = 'conda activate symphony'


class TurrealParser(SymphonyParser):

  def create_cluster(self):
    return Cluster.new('tmux')

  def setup(self):
    super().setup()
    self._setup_xmanager_client()
    self._setup_create()

  def _setup_create(self):
    parser = self.add_subparser('create', aliases=['c'])
    self._add_experiment_name(parser)
    parser.add_config_file('--network_config',
                           type=int,
                           default='liaison/configs/network/localhost.py',
                           help='number of agent pods to run in parallel.',
                           required=True)
    parser.add_argument('--results_folder',
                        '-r',
                        required=True,
                        type=str,
                        help='Results folder.')
    parser.add_argument('--experiment_name', '-n', type=str, required=True)
    parser.add_argument('--n_actors', type=int, default=1)
    self._add_dry_run(parser)

  # ==================== helpers ====================
  def _add_dry_run(self, parser):
    parser.add_argument(
        '-dr',
        '--dry-run',
        action='store_true',
        help='print the kubectl command without actually executing it.')

  def _process_experiment_name(self, experiment_name):
    """
        experiment_name will be used as DNS, so must not have underscore or dot
        """
    new_name = experiment_name.lower().replace('.', '-').replace('_', '-')
    if new_name != experiment_name:
      print('experiment name string has been fixed: {} -> {}'.format(
          experiment_name, new_name))
    return new_name

  def _network_to_component_config(self, network_config):
    """Inverts host -> component dict."""
    component_config = ConfigDict()
    for host, v in network_config.items():
      for comp in v.components:
        component_config[comp] = ConfigDict(
            host=host,
            ssh_command=v.ssh_command,
            setup_commands=v.setup_commands,
        )
    return component_config

  def _setup_xmanager_client(self, args):
    xmanager_host = args.xmanager_host
    if xmanager_host is None:
      xmanager_host = os.environ['XMANAGER_SERVER_HOST']

    xmanager_port = args.xmanager_port
    if xmanager_port is None:
      xmanager_port = os.environ['XMANAGER_SERVER_PORT']

    self._xm_client = XManagerClient(host=xmanager_host, port=xmanager_port)

  def _register_exp(self, args, network_config):

    return self._xm_client.register(name=args.experiment_name,
                                    host_name=socket.getfqdn(),
                                    results_folder=os.path.join(
                                        args.results_folder, '{exp_id}'),
                                    n_work_units=args.n_work_units,
                                    network=network_config)  # easydict works.

  def _record_commands(self, exp_id, commands):
    return self._xm_client.record_commands(exp_id=exp_id, commands=commands)

  def action_create(self, args):
    """
        Spin up a multi-node distributed Surreal experiment.
        Put any command line args that pass to the config script after "--"
    """
    network_config = ConfigDict(to_nested_dicts(args.network_config))
    component_config = self._network_to_component_config(network_config)
    cluster = self.create_cluster()
    experiment_name = self._process_experiment_name(args.experiment_name)
    exp = cluster.new_experiment(experiment_name,
                                 preamble_cmds=[ENV_ACTIVATE_CMD])

    self._setup_xmanager_client(args)
    exp_id = self._register_exp(
        args,
        network_config,
    )
    algorithm_args = args.remainder
    algorithm_args += [
        "--n_actors",
        str(args.n_actors),
    ]
    results_folder = os.path.join(args.results_folder, str(exp_id))
    print('Writing experiment output to {}'.format(results_folder))
    algorithm_args += ["--exp_id", str(exp_id)]
    algorithm_args += ["--results_folder", results_folder]
    algorithm_args += ["--network_config_file", args.network_config_file]
    executable = 'liaison/launch/main.py'
    cmd_gen = CommandGenerator(num_agents=args.num_agents,
                               num_evals=args.num_evals,
                               executable=executable,
                               config_commands=algorithm_args)

    learner = exp.new_process('learner',
                              cmds=[cmd_gen.get_command('learner')] +
                              [component_config.learner.ssh_command] +
                              component_config.learner.setup_commands)

    replay = exp.new_process('replay',
                             cmds=[cmd_gen.get_command('replay')] +
                             [component_config.replay.ssh_command] +
                             component_config.replay.setup_commands)

    ps = exp.new_process('ps',
                         cmds=[cmd_gen.get_command('ps')] +
                         [component_config.ps.ssh_command] +
                         component_config.ps.setup_commands)

    tensorboard = exp.new_process('tensorboard',
                                  cmds=[cmd_gen.get_command('tensorboard')] +
                                  [component_config.tensorboard.ssh_command] +
                                  component_config.tensorboard.setup_commands)

    tensorplex = exp.new_process('tensorplex',
                                 cmds=[cmd_gen.get_command('tensorplex')] +
                                 [component_config.tensorplex.ssh_command] +
                                 component_config.tensorplex.setup_commands)

    loggerplex = exp.new_process(
        'loggerplex',
        cmds=[cmd_gen.get_command('loggerplex')] +
        [component_config.loggerplex.ssh_command] +
        component_config.loggerplex.loggerplex.setup_commands)

    agents = []
    for i in range(args.num_agents):
      agent_name = 'agent-{}'.format(i)
      if 'agent-*' in component_config:
        key = 'agent-*'
      else:
        key = agent_name
      add_cmds = [component_config[key].ssh_command
                  ] + component_config[key].setup_commands
      agent = exp.new_process(agent_name,
                              cmds=[cmd_gen.get_command(agent_name)] +
                              add_cmds)
      agents.append(agent)

    setup_network(agents=agents,
                  learner=learner,
                  replay=replay,
                  ps=ps,
                  tensorboard=tensorboard,
                  tensorplex=tensorplex,
                  loggerplex=loggerplex)
    cluster.launch(exp, dry_run=args.dry_run)


def main():
  TurrealParser().main()


if __name__ == '__main__':
  main()
