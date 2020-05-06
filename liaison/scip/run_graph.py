# Evaluate on a single MIP instance using the environment and the agent
# as primal heuristic in SCIP.
import os
import pickle
from pathlib import Path

import liaison.utils as U
from absl import app, logging
from argon import ArgumentParser, to_nested_dicts
from liaison.scip.evaluate import Evaluator
from liaison.scip.scip_integration import (EvalHeur, init_scip_params,
                                           run_branch_and_bound_scip)
from liaison.utils import ConfigDict
from pyscipopt import Model

parser = ArgumentParser()
# agent_config
parser.add_config_file(name='agent', required=True)

# env_config
parser.add_config_file(name='env', required=True)

# sess_config
parser.add_config_file(name='sess', required=True)

parser.add_argument('--n_local_moves', type=int, required=True)

parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gap', type=float, default=.0)
parser.add_argument('--max_nodes', type=int)
parser.add_argument('-n', '--name', required=True)
parser.add_argument('--use_parallel_envs', action='store_true')
parser.add_argument('--use_threaded_envs', action='store_true')
parser.add_argument('--without_scip', action='store_true')
parser.add_argument('--without_agent', action='store_true')
parser.add_argument('--gpu_ids', '-g', type=int, nargs='+')
parser.add_argument('--heuristic', type=str)
parser.add_argument('--heur_frequency',
                    type=int,
                    default=-1,
                    help='Use -1 to completely turn off heuristics.')
args = None


def main(argv):
  global args
  args = parser.parse_args(argv[1:])
  if args.gpu_ids:
    os.environ['CUDA_VISIBLE_DEVICES'] = '_'.join(map(str, args.gpu_ids))
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

  sess_config = ConfigDict(to_nested_dicts(args.sess_config))
  env_config = ConfigDict(to_nested_dicts(args.env_config))
  agent_config = ConfigDict(to_nested_dicts(args.agent_config))

  shell_class = U.import_obj(sess_config.shell.class_name, sess_config.shell.class_path)
  env_class = U.import_obj(env_config.class_name, env_config.class_path)

  agent_class = U.import_obj(agent_config.class_name, agent_config.class_path)

  if args.without_scip:
    results_dir = Path(
        f'/data/nms/tfp/evaluation/without_scip/{args.name}/{env_config.graph_start_idx}/')
  elif args.without_agent:
    results_dir = Path(
        f'/data/nms/tfp/evaluation/without_agent/{args.name}/{env_config.graph_start_idx}/')
  elif args.heuristic:
    results_dir = Path(
        f'/data/nms/tfp/evaluation/{args.heuristic}/{args.name}/{env_config.graph_start_idx}/')
  else:
    results_dir = Path(f'/data/nms/tfp/evaluation/scip/{args.name}/{env_config.graph_start_idx}')
  results_dir.mkdir(parents=True, exist_ok=True)

  evaluator = Evaluator(shell_class=shell_class,
                        shell_config=sess_config.shell,
                        agent_class=agent_class,
                        agent_config=agent_config,
                        env_class=env_class,
                        env_config=env_config,
                        seed=args.seed,
                        dataset=env_config.dataset,
                        dataset_type=env_config.dataset_type,
                        graph_start_idx=env_config.graph_start_idx,
                        gap=args.gap,
                        max_nodes=args.max_nodes,
                        batch_size=args.batch_size,
                        n_local_moves=args.n_local_moves,
                        results_dir=results_dir,
                        use_parallel_envs=args.use_parallel_envs,
                        use_threaded_envs=args.use_threaded_envs,
                        heur_frequency=args.heur_frequency,
                        **sess_config)
  evaluator.run(without_scip=args.without_scip,
                without_agent=args.without_agent,
                heuristic=args.heuristic)
  print('Done!')


if __name__ == "__main__":
  app.run(main)
