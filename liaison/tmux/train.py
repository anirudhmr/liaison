# python liaison/tmux/train.py --pdb_post_mortem -- create -r /data/nms/tfp/results/{experiment_name}/{exp_id} -e gn --filter='.*freebsd_([0-9]|1[0-5])_c220g5.*|.*110533.*' --n_actors=8 --resource_req_config.actor.mem=175 -- --agent_config_file=liaison/configs/agent/gcn.py --sess_config_file=liaison/configs/session_config.py --env_config_file=liaison/configs/env_config.py --env_config.class_path='liaison.env.shortest_path' --sess_config.learner.use_gpu=True --batch_size=128 --sess_config.learner.batch_size=32 --sess_config.actor.use_threaded_envs=False --traj_length=128 ^Cagent_config.discount_factor=1. --sess_config.replay.memory_size=8000 --sess_config.replay.load_balanced=False --sess_config.learner.publish_every=50 --agent_config.grad_clip=0.0 --agent_config.model.n_prop_layers=8 --agent_config.lr_min=1e-4 --sess_config.learner.inmem_tmp_dir='/users/arc/vol/work_dir'
# python liaison/tmux/train.py --pdb_post_mortem -- create -r /tmp/ -e test --filter='.*os_mm.*' -- --agent_config_file=liaison/configs/agent/config.py --sess_config_file=liaison/configs/session_config.py --env_config_file=liaison/configs/env_config.py
import argparse
import json
import shlex

import argon
from absl import app
from liaison.launch import hyper
from liaison.tmux.create_programs import build_program
from liaison.tmux.liaison_placer import LiaisonPlacer
from liaison.tmux.surreal_tmux import TurrealParser
from liaison.utils import ConfigDict

IRS_NODE = 'lincoln_supercloud_control'
IRS_PROXY_NODE = 'lincoln_supercloud_control'
VISUALIZER_NODE = 'cloudlab_clemson_clgpu005'
WHITELIST_NODES = list(set((IRS_NODE, IRS_PROXY_NODE, VISUALIZER_NODE)))

parser = argon.ArgumentParser('Liaison trainer', add_help=False)
parser.add_argument('--n_actors', type=int, default=1)
parser.add_argument('--bundle_actors', action='store_true')
parser.add_config_file(name='cluster', default='ccc/config.py')
parser.add_config_file(name='resource_req', default='liaison/configs/resource_req.py')
parser.add_argument('--spy_measurement_interval', type=float, default=2.)
parser.add_argument('--gpu_overload_obj_coeff', type=float, default=1)
parser.add_argument('--gpu_load_balancing_obj_coeff', type=float, default=1)
parser.add_argument('--gpu_wu_consolidation_obj_coeff', type=float, default=.25)
parser.add_argument('--cpu_overload_obj_coeff', type=float, default=1)
parser.add_argument('--cpu_load_balancing_obj_coeff', type=float, default=1)
parser.add_argument('--cpu_wu_consolidation_obj_coeff', type=float, default=10)
parser.add_argument('--filter_nodes_regex', type=str, default='.*')
parser.add_argument('--without_evaluators', action='store_true')
parser.add_argument('--without_valid_and_test_evaluators', action='store_true')
parser.add_argument('--without_visualizers', action='store_true')
parser.add_argument(
    '--whitelist_nodes',
    nargs='+',
    default=WHITELIST_NODES,
    help='These nodes are always selected irrespective of the filter_nodes_regex specified.')
# placement constraints
parser.add_argument('--pl_constraints',
                    type=str,
                    nargs='+',
                    default=[],
                    help='''
  If you would like to place all actor nodes on gpu-* servers, then use the string:
  'actor-.*:gpu-.*'
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
parser.add_argument('--disable_sweep', '--no_sweep', '--without_sweep', action='store_true')
parser.add_argument('--slurm_colocate_wunit', action='store_true')
parser.add_argument('--slurm_per_gpu_allocation', action='store_true')
parser.add_argument('--use_irs_proxy', action='store_true')


def validate_args(args):
  if args.bundle_actors:
    # use sess_config.actor_bundle.n_actors in this case.
    assert args.n_actors == 1


def train(argv):
  tp = TurrealParser()
  tp.add_external_parser(parser)
  func, external_parser_args = tp.main(argv[1:])
  if func.__name__.split('action_')[-1] != 'create':
    return
  args = external_parser_args[0]
  validate_args(args)

  cluster = tp.get_cluster()

  # Specify experiment specific flags here.
  exp_flags = []
  hyper_configs = []
  exps = []
  for work_id, params in enumerate(
      hyper.product(
          # hyper.discrete('env_config.k', [10, 20]),
          # hyper.discrete('agent_config.lr_init', [2e-4]),
          hyper.discrete('agent_config.lr_init', [1e-4, 2e-4, 3e-4, 4e-4]),
          hyper.discrete('agent_config.ent_dec_init', [2e-2]),
          # hyper.discrete('env_config.graph_start_idx', list(range(8))),
      )):
    exp = cluster.new_experiment('%s-%d' % (tp.experiment_name, work_id), env_name='liaison')
    # start tensorboard only for the first work unit.
    coloc_constraints = build_program(
        exp,
        args.n_actors,
        ConfigDict(argon.to_nested_dicts(args.resource_req_config)),
        bundle_actors=args.bundle_actors,
        irs_placement=IRS_NODE,
        visualizer_placement=VISUALIZER_NODE,
        with_visualizers=(work_id == 0) and (not args.without_visualizers),
        with_evaluators=(not args.without_evaluators),
        without_valid_and_test_evaluators=args.without_valid_and_test_evaluators,
        with_irs_proxy=args.use_irs_proxy,
        irs_proxy_placement=IRS_PROXY_NODE)

    exp_flag = ['--work_id', str(work_id)]
    exp_flag += ['--hyper_configs', str(shlex.quote(json.dumps(params)))]
    exp_flag += hyper.to_commandline(params)
    exps.append(exp)
    exp_flags.append(exp_flag)
    hyper_configs.append(params)
    if args.disable_sweep:
      break

  exp_procs = [[proc for pg in exp.list_process_groups()
                for proc in pg.list_processes()] + [proc for proc in exp.list_processes()]
               for exp in exps]
  print('-----------exp stats-------------')
  print('Number of work units: %d' % len(exps))
  print('Number of processes total: %d' % sum(map(len, exp_procs)))

  placer = LiaisonPlacer(
      tp.exp_id,
      exps,
      ConfigDict(argon.to_nested_dicts(args.cluster_config)),
      args.filter_nodes_regex,
      args.whitelist_nodes,
      args.spy_measurement_interval,
      pl_constraints=list(map(lambda k: k.split(':'), args.pl_constraints)),
      coloc_constraints=list(
          map(lambda k: k.split(';'), coloc_constraints + args.coloc_constraints)),
      gpu_overload_obj_coeff=args.gpu_overload_obj_coeff,
      gpu_load_balancing_obj_coeff=args.gpu_load_balancing_obj_coeff,
      gpu_wu_consolidation_obj_coeff=args.gpu_wu_consolidation_obj_coeff,
      cpu_overload_obj_coeff=args.cpu_overload_obj_coeff,
      cpu_load_balancing_obj_coeff=args.cpu_load_balancing_obj_coeff,
      cpu_wu_consolidation_obj_coeff=args.cpu_wu_consolidation_obj_coeff,
      slurm_colocate_wunit=args.slurm_colocate_wunit,
      slurm_per_gpu_allocation=args.slurm_per_gpu_allocation,
  )

  tp.launch(exps, exp_flags, hyper_configs)


if __name__ == '__main__':
  app.run(train)
