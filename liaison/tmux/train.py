import argparse

from liaison.launch import hyper
from liaison.tmux.surreal_tmux import TurrealParser
from liaison.tmux.liaison_placer import LiaisonPlacer
from liaison.tmux.create_programs import build_program
from liaison.utils import ConfigDict
import argon
from absl import app

parser = argon.ArgumentParser('Resource Requirement config', add_help=False)
parser.add_argument('--n_actors', type=int, default=1)
parser.add_config_file(name='resource_req',
                       default='liaison/configs/resource_req.py')
parser.add_argument('--spy_measurement_interval', type=float, default=2.)
parser.add_argument('--gpu_overload_obj_coeff', type=int, default=1)
parser.add_argument('--gpu_load_balancing_obj_coeff', type=int, default=1)
parser.add_argument('--gpu_wu_consolidation_obj_coeff', type=int, default=.25)
parser.add_argument('--cpu_overload_obj_coeff', type=int, default=1)
parser.add_argument('--cpu_load_balancing_obj_coeff', type=int, default=1)
parser.add_argument('--cpu_wu_consolidation_obj_coeff', type=int, default=10)

# placement constraints
parser.add_argument('--pl_constraints',
                    type=str,
                    nargs='+',
                    default=[],
                    help='''
  If you would like to place all actor nodes on gpu-* servers, then use the string:
  'actor-*:gpu-*'
  ''')
parser.add_argument('--coloc_constraints',
                    type=str,
                    nargs='+',
                    default=[],
                    help='''
  If you have the following three coloc constraints:
    (a, b, c), (p, q, r), (x, y, z)
    then use the string:
      a;b;c p;q;r x;y;z
  ''')


def train(argv):
  tp = TurrealParser()
  tp.add_external_parser(parser)
  func, external_parser_args = tp.main(argv[1:])
  if func.__name__.split('action_')[-1] != 'create':
    return
  args = external_parser_args[0]

  cluster = tp.get_cluster()
  nodes = tp.get_nodes()

  # Specify experiment specific flags here.
  exp_flags = []
  exps = []
  for work_id, params in enumerate(
      hyper.discrete('agent_config.lr_init', [1e-3])):
    exp = cluster.new_experiment('%s-%d' % (tp.experiment_name, work_id),
                                 env_name='liaison')
    build_program(exp, args.n_actors,
                  ConfigDict(argon.to_nested_dicts(args.resource_req_config)))

    exp_flag = ['--work_id', str(work_id)]
    exp_flag += ["--n_actors", str(args.n_actors)]
    exp_flag += hyper.to_commandline(params)
    exps.append(exp)
    exp_flags.append(exp_flag)

  exp_procs = [[
      proc for pg in exp.list_process_groups() for proc in pg.list_processes()
  ] + [proc for proc in exp.list_processes()] for exp in exps]
  print('-----------exp stats-------------')
  print('Number of work units: %d' % len(exps))
  print('Number of processes total: %d' % sum(map(len, exp_procs)))

  placer = LiaisonPlacer(
      exps,
      nodes,
      args.spy_measurement_interval,
      pl_constraints=list(map(lambda k: k.split(':'), args.pl_constraints)),
      coloc_constraints=list(
          map(lambda k: k.split(';'), args.coloc_constraints)),
      gpu_overload_obj_coeff=args.gpu_overload_obj_coeff,
      gpu_load_balancing_obj_coeff=args.gpu_load_balancing_obj_coeff,
      gpu_wu_consolidation_obj_coeff=args.gpu_wu_consolidation_obj_coeff,
      cpu_overload_obj_coeff=args.cpu_overload_obj_coeff,
      cpu_load_balancing_obj_coeff=args.cpu_load_balancing_obj_coeff,
      cpu_wu_consolidation_obj_coeff=args.cpu_wu_consolidation_obj_coeff)

  tp.launch(exps, exp_flags)


if __name__ == '__main__':
  app.run(train)
  # train(sys.argv[1])
