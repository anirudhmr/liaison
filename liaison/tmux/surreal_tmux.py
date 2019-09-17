import math
import os
import shlex
import socket
import sys
from copy import copy

import liaison.utils as U
from argon import ArgumentParser, to_nested_dicts
from liaison.launch import CommandGenerator, setup_network
from liaison.launch.xmanager_client import XManagerClient
from liaison.utils import ConfigDict, ConfigDict_to_dict
from symphony.commandline import SymphonyParser
from symphony.engine import Cluster

ENV_ACTIVATE_CMD = 'conda activate symphony'
PYTHONPATH_CMD = 'export PYTHONPATH="$PYTHONPATH:`pwd`"'
PREAMBLE_CMDS = [ENV_ACTIVATE_CMD, PYTHONPATH_CMD]


class TurrealParser(SymphonyParser):

  def create_cluster(self):
    return Cluster.new('tmux')

  def setup(self):
    super().setup()
    self._setup_create()

  def _setup_create(self):
    parser = self.add_subparser('create', aliases=['c'])
    self._add_experiment_name(parser, positional=False)
    parser.add_argument('--results_folder',
                        '-r',
                        required=True,
                        type=str,
                        help='Results folder.')
    parser.add_argument('--n_work_units', type=int, default=1)
    parser.add_argument('--n_actors', type=int, default=1)
    self._add_dry_run(parser)

    # Note we cannot add it as a  subparser since argon doesn't support this.
    # Hence we resort to two different sequential parsers
    # see main function below.
    network_config_parser = ArgumentParser('Network config', add_help=False)
    network_config_parser.add_config_file(
        name='network',
        default='liaison/configs/network/localhost.py',
        help='number of actor pods to run in parallel.')
    self._network_config_parser = network_config_parser

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
      if isinstance(v, dict):  # ignore --network_config_file
        print(host, v)
        for comp in v.components:
          component_config[comp] = ConfigDict(
              host=host,
              ssh_command=v.ssh_command,
              setup_commands=v.setup_commands,
          )
    return component_config

  def _setup_xmanager_client(self, args):

    self._xm_client = XManagerClient(host=args.xmanager_server_host,
                                     port=int(args.xmanager_server_port))

  def _register_exp(self, args, results_folder, network_config):

    return self._xm_client.register(name=args.experiment_name,
                                    host_name=socket.getfqdn(),
                                    results_folder=results_folder,
                                    n_work_units=args.n_work_units,
                                    network=ConfigDict_to_dict(network_config),
                                    dry_run=args.dry_run)  # easydict works.

  def _record_commands(self, exp_id, preamble_cmds, procs, dry_run):
    commands = dict()
    for proc in procs:
      env_cmds = [
          'export {}={}'.format(k, shlex.quote(v))
          for k, v in proc.env.items()
      ]
      cmds = env_cmds + preamble_cmds + proc.cmds
      commands[proc.name] = cmds

    return self._xm_client.record_commands(exp_id=exp_id,
                                           commands=commands,
                                           dry_run=dry_run)

  def action_create(self, args):
    """
        Spin up a multi-node distributed Surreal experiment.
        Put any command line args that pass to the config script after "--"
    """
    network_config = self._network_config_args.network_config
    network_config = ConfigDict(to_nested_dicts(network_config))
    component_config = self._network_to_component_config(network_config)
    results_folder = args.results_folder
    if '{experiment_name}' in results_folder:
      results_folder = results_folder.format(
          experiment_name=args.experiment_name)
    cluster = self.create_cluster()
    experiment_name = self._process_experiment_name(args.experiment_name)
    exp = cluster.new_experiment(experiment_name, preamble_cmds=PREAMBLE_CMDS)

    self._setup_xmanager_client(args)
    exp_id = self._register_exp(
        args,
        results_folder,
        network_config,
    )
    algorithm_args = args.remainder
    algorithm_args += [
        "--n_actors",
        str(args.n_actors),
    ]
    if '{exp_id}' in results_folder:
      results_folder = results_folder.format(exp_id=exp_id)
    print('Experiment ID: %d' % exp_id)
    print('Results folder: %s' % (results_folder))
    algorithm_args += ["--experiment_id", str(exp_id)]
    algorithm_args += ["--experiment_name", experiment_name]
    algorithm_args += ["--work_id", str(0)]
    algorithm_args += ["--results_folder", results_folder]
    algorithm_args += [
        "--network_config_file", self._network_config_args.network_config_file
    ]
    executable = 'liaison/launch/main.py'
    cmd_gen = CommandGenerator(executable=executable,
                               config_commands=algorithm_args)

    learner = exp.new_process('learner',
                              cmds=[component_config.learner.ssh_command] +
                              component_config.learner.setup_commands +
                              [cmd_gen.get_command('learner')])

    replay = exp.new_process('replay',
                             cmds=[component_config.replay.ssh_command] +
                             component_config.replay.setup_commands +
                             [cmd_gen.get_command('replay')])

    ps = exp.new_process('ps',
                         cmds=[component_config.ps.ssh_command] +
                         component_config.ps.setup_commands +
                         [cmd_gen.get_command('ps')])

    tensorboard = exp.new_process(
        'tensorboard',
        cmds=[component_config.tensorboard.ssh_command] +
        component_config.tensorboard.setup_commands +
        [cmd_gen.get_command('tensorboard')])

    tensorplex = exp.new_process(
        'tensorplex',
        cmds=[component_config.tensorplex.ssh_command] +
        component_config.tensorplex.setup_commands +
        [cmd_gen.get_command('tensorplex')])

    loggerplex = exp.new_process(
        'loggerplex',
        cmds=[component_config.loggerplex.ssh_command] +
        component_config.loggerplex.setup_commands +
        [cmd_gen.get_command('loggerplex')])

    actors = []
    for i in range(args.n_actors):
      actor_name = 'actor-{}'.format(i)
      if 'actor-*' in component_config:
        key = 'actor-*'
      else:
        key = actor_name
      add_cmds = [component_config[key].ssh_command
                  ] + component_config[key].setup_commands
      actor = exp.new_process(actor_name,
                              cmds=add_cmds +
                              [cmd_gen.get_command(actor_name)])
      actors.append(actor)

    setup_network(
        actors=actors,
        learner=learner,
        replay=replay,
        ps=ps,
        tensorboard=tensorboard,
        tensorplex=tensorplex,
        loggerplex=loggerplex,
    )

    self._record_commands(exp_id,
                          exp.preamble_cmds,
                          procs=[
                              *actors, learner, replay, ps, tensorboard,
                              tensorplex, loggerplex
                          ],
                          dry_run=args.dry_run)
    cluster.launch(exp, dry_run=args.dry_run)

  def main(self):
    assert sys.argv.count('--') <= 1, \
        'command line can only have at most one "--"'
    if '--' in sys.argv:
      idx = sys.argv.index('--')
      remainder = sys.argv[idx + 1:]
      sys.argv = sys.argv[:idx]
      has_remainder = True  # even if remainder itself is empty
    else:
      remainder = []
      has_remainder = False

    # note subparser cannot be argon.ArgumentParser
    # so we prune out the network config arguments first before passing the rest
    # through the main parser.
    self._network_config_args, unknown = self._network_config_parser.parse_known_args(
    )
    master_args = unknown[1:]
    if '--' in master_args:
      master_args = master_args[:master_args.index('--')]
    args = self.master_parser.parse_args(master_args)
    args.remainder = remainder
    args.has_remainder = has_remainder

    args.func(args)


def main():
  TurrealParser().main()


if __name__ == '__main__':
  main()
