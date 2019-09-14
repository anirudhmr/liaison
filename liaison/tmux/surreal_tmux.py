import math
import os
import sys
from copy import copy

import liaison.utils as U
from benedict import BeneDict
from liaison.launch import CommandGenerator, setup_network
from symphony.commandline import SymphonyParser
from symphony.engine import Cluster, SymphonyConfig


class TurrealParser(SymphonyParser):

  def create_cluster(self):
    return Cluster.new('tmux')

  def setup(self):
    super().setup()
    self._setup_create()

  @property
  def folder(self):
    if 'tmux_results_folder' not in self.config:
      raise KeyError('Please specify "tmux_results_folder" in ~/.surreal.yml')
    return U.f_expand(self.config.tmux_results_folder)

  @property
  def username(self):
    if 'username' not in self.config:
      raise KeyError('Please specify "username" in ~/.surreal.yml')
    return self.config.username

  def _setup_create(self):
    parser = self.add_subparser('create', aliases=['c'])
    self._add_experiment_name(parser)
    parser.add_argument('--num_agents',
                        type=int,
                        default=2,
                        help='number of agent pods to run in parallel.')
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

  def action_create(self, args):
    """
            Spin up a multi-node distributed Surreal experiment.
            Put any command line args that pass to the config script after "--"
        """
    cluster = self.create_cluster()
    experiment_name = self._process_experiment_name(args.experiment_name)
    exp = cluster.new_experiment(experiment_name,
                                 preamble_cmds=self.config.tmux_preamble_cmds)

    algorithm_args = args.remainder
    algorithm_args += [
        "--num-agents",
        str(args.num_agents),
    ]
    experiment_folder = os.path.join(self.folder, experiment_name)
    print('Writing experiment output to {}'.format(experiment_folder))
    algorithm_args += ["--experiment-folder", experiment_folder]
    algorithm_args += ["--env", args.env]
    executable = 'liaison/launch/main.py'
    cmd_gen = CommandGenerator(num_agents=args.num_agents,
                               num_evals=args.num_evals,
                               executable=executable,
                               config_commands=algorithm_args)

    learner = exp.new_process('learner', cmds=[cmd_gen.get_command('learner')])

    replay = exp.new_process('replay', cmds=[cmd_gen.get_command('replay')])

    ps = exp.new_process('ps', cmds=[cmd_gen.get_command('ps')])

    tensorboard = exp.new_process('tensorboard',
                                  cmds=[cmd_gen.get_command('tensorboard')])

    tensorplex = exp.new_process('tensorplex',
                                 cmds=[cmd_gen.get_command('tensorplex')])

    loggerplex = exp.new_process('loggerplex',
                                 cmds=[cmd_gen.get_command('loggerplex')])

    agents = []
    for i in range(args.num_agents):
      agent_name = 'agent-{}'.format(i)
      agent = exp.new_process(agent_name,
                              cmds=[cmd_gen.get_command(agent_name)])
      agents.append(agent)

    evals = []
    for i in range(args.num_evals):
      eval_name = 'eval-{}'.format(i)
      eval_p = exp.new_process(eval_name,
                               cmds=[cmd_gen.get_command(eval_name)])
      evals.append(eval_p)

    setup_network(agents=agents,
                  evals=evals,
                  learner=learner,
                  replay=replay,
                  ps=ps,
                  tensorboard=tensorboard,
                  tensorplex=tensorplex,
                  loggerplex=loggerplex)
    # self._setup_gpu(agents=agents, evals=evals, learner=learner, gpus=args.gpu)
    cluster.launch(exp, dry_run=args.dry_run)

  def _setup_gpu(self, agents, evals, learner, gpus):
    """
            Assigns GPU to agents and learners in an optimal way.
            No GPU, do nothing
        """
    actors = agents + evals
    if gpus == "auto":
      if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpus = os.environ['CUDA_VISIBLE_DEVICES']
      else:
        gpus = ''
    gpus_str = gpus
    try:
      gpus = [x for x in gpus_str.split(',') if len(x) > 0]
    except Exception as e:
      print("Error parsing GPU specification {}\n".format(gpus_str),
            file=sys.stderr)
      raise e
    if len(gpus) == 0:
      print('Using CPU')
    elif len(gpus) == 1:
      gpu = gpus[0]
      print('Putting agents, evals and learner on GPU {}'.format(gpu))
      for actor in actors + [learner]:
        actor.set_envs({'CUDA_VISIBLE_DEVICES': gpu})
    elif len(gpus) > 1:
      learner_gpu = gpus[0]
      print('Putting learner on GPU {}'.format(learner_gpu))
      learner.set_envs({'CUDA_VISIBLE_DEVICES': learner_gpu})

      actors_per_gpu = float(len(actors)) / (len(gpus) - 1)
      actors_per_gpu = int(math.ceil(actors_per_gpu))
      print('Putting up to {} agents/evals on each of gpus {}'.format(
          actors_per_gpu, ','.join([x for x in gpus[1:]])))

      for i, actor in enumerate(actors):
        cur_gpu = gpus[1 + i // actors_per_gpu]
        actor.set_envs({'CUDA_VISIBLE_DEVICES': cur_gpu})


def main():
  TurrealParser().main()


if __name__ == '__main__':
  main()
