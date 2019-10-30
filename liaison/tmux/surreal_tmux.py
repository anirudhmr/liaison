import math
import os
import shlex
import socket
import sys
from copy import copy
from multiprocessing.pool import ThreadPool

import liaison.utils as U
from argon import ArgumentParser, to_nested_dicts
from caraml.zmq import ZmqClient
from liaison.launch import CommandGenerator
from liaison.launch.xmanager_client import get_xmanager_client
from liaison.utils import ConfigDict, ConfigDict_to_dict
from symphony.commandline import SymphonyParser
from symphony.engine import Cluster
from symphony.tmux import Node
from ccc.src import NodeLoader

ENV_ACTIVATE_CMD = 'conda activate liaison'
PYTHONPATH_CMD = 'export PYTHONPATH="$PYTHONPATH:`pwd`"'
PREAMBLE_CMDS = [ENV_ACTIVATE_CMD, PYTHONPATH_CMD]

RES_DIRS = [
    '/'.join(__file__.split('/')[:-2]),
    '/'.join(__file__.split('/')[:-3]) + '/.git'
]


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
    parser.add_argument('--filter_nodes_regex', type=str, default='.*')
    self._add_dry_run(parser)

    # Note we cannot add it as a  subparser since argon doesn't support this.
    # Hence we resort to two different sequential parsers
    # see main function below.
    cluster_config_parser = ArgumentParser('Cluster config', add_help=False)
    cluster_config_parser.add_config_file(name='cluster',
                                          default='ccc/config.py')
    self._cluster_config_parser = cluster_config_parser

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

  def _setup_xmanager_client(self, args):

    self._xm_client = get_xmanager_client(host=args.xmanager_server_host,
                                          port=int(args.xmanager_server_port),
                                          timeout=4)

  def _register_exp(self, exp_name):
    return self._xm_client.register(name=exp_name)

  def _record_commands(self, exp_id, preamble_cmds, procs, dry_run):
    commands = dict()
    for proc in procs:
      env_cmds = [
          'export {}={}'.format(k, shlex.quote(v))
          for k, v in proc.env.items()
      ]
      cmds = env_cmds + preamble_cmds + proc.cmds
      commands[proc.name] = cmds

    self._xm_client.record_commands(exp_id=exp_id,
                                    commands=commands,
                                    dry_run=dry_run)
    return commands

  def _register_commands_with_irs(self, commands, host, port):
    self._cli = ZmqClient(host=host,
                          port=port,
                          serializer='pyarrow',
                          deserializer='pyarrow')
    self._cli.request(['register_commands', [], commands])

  def _make_nodes(self, cluster_config, args):
    nodes = NodeLoader(cluster_config, args.filter_nodes_regex).nodes
    print('Filtered nodes (%d): ' % len(nodes))
    for node in nodes:
      print(node.name)
    with ThreadPool(len(nodes)) as pool:
      pool.map(lambda node: node.setup(res_dirs=RES_DIRS), nodes)
    return nodes

  def get_cluster(self):
    return self.cluster

  def get_nodes(self):
    return self.nodes

  def action_create(self, args):
    """
        Spin up a multi-node distributed Surreal experiment.
        Put any command line args that pass to the config script after "--"
    """
    self.experiment_name = self._process_experiment_name(args.experiment_name)
    results_folder = args.results_folder
    if '{experiment_name}' in results_folder:
      results_folder = results_folder.format(
          experiment_name=self.experiment_name)
    self.cluster = cluster = self.create_cluster()
    self._setup_xmanager_client(args)

    cluster_config = ConfigDict(
        to_nested_dicts(self._cluster_config_args.cluster_config))

    nodes = self._make_nodes(cluster_config, args)
    exp_id = self._register_exp(self.experiment_name)
    if '{exp_id}' in results_folder:
      results_folder = results_folder.format(exp_id=exp_id)

    self.results_folder = results_folder
    self.nodes = nodes
    self.remainder_args = args.remainder
    self.exp_id = exp_id

  def launch(self, experiments, exp_configs):
    """
    Tasks:
      1. Creates nodes
        1.1 Provide them with spy server addresses to get load details.
        1.2 Sets up nodes (file systems, libraries, resources etc.)

      2. Adds all the commands needed for processes, shells and experiments.

      3. Register with XManager client and IRS

      4. Launch experiments

    Details:
      1. Add PREAMBLE_CMDS to the experiment
    """

    exp_id = self.exp_id
    print('Experiment ID: %d' % exp_id)
    print('Results folder: %s' % (self.results_folder))
    algorithm_args = self.remainder_args
    algorithm_args += ["--experiment_id", str(exp_id)]
    algorithm_args += ["--experiment_name", self.experiment_name]
    algorithm_args += ["--results_folder", self.results_folder]

    commands = []
    for exp, exp_config in zip(experiments, exp_configs):
      cmd_gen = CommandGenerator(executable='liaison/launch/main.py',
                                 config_commands=algorithm_args + exp_config)

      exp.set_preamble_cmds(PREAMBLE_CMDS)
      all_procs = [
          proc for pg in exp.list_process_groups()
          for proc in pg.list_processes()
      ] + [proc for proc in exp.list_processes()]

      for proc in all_procs:
        proc.append_cmds([cmd_gen.get_command(proc.name)])

      for proc in all_procs:
        commands.append(cmd_gen.get_command(proc.name))

      self.cluster.launch(exp)
    # self._register_commands_with_irs(commands,
    #                                  host=irs.env['SYMPH_IRS_FRONTEND_HOST'],
    #                                  port=irs.env['SYMPH_IRS_FRONTEND_PORT'])

  def main(self, argv):
    assert argv.count('--') <= 1, \
        'command line can only have at most one "--"'

    argv = list(argv)
    if '--' in argv:
      idx = argv.index('--')
      remainder = argv[idx + 1:]
      argv = argv[:idx]
      has_remainder = True  # even if remainder itself is empty
    else:
      remainder = []
      has_remainder = False
    master_args = argv

    # note subparser cannot be argon.ArgumentParser
    # so we prune out the network config arguments first before passing the rest
    # through the main parser.
    # Also we prune out the external parser arguments as well.
    args_l = []
    for parser in [self._cluster_config_parser] + self._external_parsers:
      args, unknown = parser.parse_known_args(master_args)
      master_args = unknown
      args_l.append(args)

    self._cluster_config_args = args_l[0]
    assert '--' not in master_args
    args = self.master_parser.parse_args(master_args)
    args.remainder = remainder
    args.has_remainder = has_remainder

    args.func(args)
    return args.func, args_l[1:]
