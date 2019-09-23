import math
import os
import shlex
import socket
import sys
from copy import copy

import liaison.utils as U
from argon import ArgumentParser, to_nested_dicts
from caraml.zmq import ZmqClient
from liaison.launch import CommandGenerator, setup_network
from liaison.launch.xmanager_client import XManagerClient
from liaison.utils import ConfigDict, ConfigDict_to_dict
from symphony.commandline import SymphonyParser
from symphony.engine import Cluster
from symphony.tmux import Node

ENV_ACTIVATE_CMD = 'conda activate symphony'
PYTHONPATH_CMD = 'export PYTHONPATH="$PYTHONPATH:`pwd`"'
PREAMBLE_CMDS = [ENV_ACTIVATE_CMD, PYTHONPATH_CMD]

RESOURCES = {
    'file': [],
    'dir': [
        'liaison/',
        'caraml/',
    ]
}


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

  def _get_nodes(self, network_config):
    nodes = []
    component2node = dict()

    if 'host_names' in network_config:
      host_names_to_ip = network_config.host_names
    else:
      # TODO: Use default host names file here.
      host_names_to_ip = {}

    for host_name, node_config in network_config.hosts.items():
      node = Node(host_names_to_ip[host_name], **node_config)
      for comp in node_config.components:
        component2node[comp] = node
      nodes.append(node)

    return nodes, component2node

  def _setup_nodes(self, nodes):
    for node in nodes:
      if node.use_ssh:
        for fname in RESOURCES[
            'file']:  # should be relative to liaison directory
          node.put_file(U.f_expand(fname), os.path.join(node.base_dir, fname))
        for dirname in RESOURCES['dir']:
          node.put_dir(U.f_expand(dirname),
                       os.path.join(node.base_dir, dirname))

  def action_create(self, args):
    """
        Spin up a multi-node distributed Surreal experiment.
        Put any command line args that pass to the config script after "--"
    """
    network_config = self._network_config_args.network_config
    network_config = ConfigDict(to_nested_dicts(network_config))
    nodes, component_to_node = self._get_nodes(network_config)
    self._setup_nodes(nodes)
    sys.exit(0)
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
    wid = 0
    if '{exp_id}' in results_folder:
      results_folder = results_folder.format(exp_id=exp_id)

    print('Experiment ID: %d' % exp_id)
    print('Results folder: %s' % (results_folder))
    algorithm_args += ["--experiment_id", str(exp_id)]
    algorithm_args += ["--experiment_name", experiment_name]
    algorithm_args += ["--work_id", str(wid)]
    algorithm_args += ["--results_folder", results_folder]
    algorithm_args += [
        "--network_config_file", self._network_config_args.network_config_file
    ]
    executable = 'liaison/launch/main.py'
    cmd_gen = CommandGenerator(executable=executable,
                               config_commands=algorithm_args)

    learner = exp.new_process('learner',
                              component_to_node['learner'],
                              cmds=[cmd_gen.get_command('learner')])

    replay = exp.new_process('replay',
                             component_to_node['replay'],
                             cmds=[cmd_gen.get_command('replay')])

    ps = exp.new_process('ps',
                         component_to_node['ps'],
                         cmds=[cmd_gen.get_command('ps')])

    irs = exp.new_process('irs',
                          component_to_node['irs'],
                          cmds=[cmd_gen.get_command('irs')])

    tensorboard = exp.new_process('tensorboard',
                                  component_to_node['tensorboard'],
                                  cmds=[cmd_gen.get_command('tensorboard')])

    tensorplex = exp.new_process('tensorplex',
                                 component_to_node['tensorplex'],
                                 cmds=[cmd_gen.get_command('tensorplex')])

    actors = []
    for i in range(args.n_actors):
      actor_name = 'actor-{}'.format(i)
      if 'actor-*' in component_to_node:
        key = 'actor-*'
      else:
        key = actor_name
      actor = exp.new_process(actor_name,
                              component_to_node[key],
                              cmds=[cmd_gen.get_command(actor_name)])
      actors.append(actor)

    setup_network(
        actors=actors,
        learner=learner,
        replay=replay,
        ps=ps,
        tensorboard=tensorboard,
        tensorplex=tensorplex,
        irs=irs,
    )

    commands = self._record_commands(exp_id,
                                     exp.preamble_cmds,
                                     procs=[
                                         *actors,
                                         learner,
                                         replay,
                                         ps,
                                         tensorboard,
                                         tensorplex,
                                         irs,
                                     ],
                                     dry_run=args.dry_run)
    cluster.launch(exp, dry_run=args.dry_run)
    self._register_commands_with_irs(commands,
                                     host=irs.env['SYMPH_IRS_FRONTEND_HOST'],
                                     port=irs.env['SYMPH_IRS_FRONTEND_PORT'])

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
