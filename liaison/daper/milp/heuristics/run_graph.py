import os
import pickle
import sys
from pathlib import Path

from absl import app
from argon import ArgumentParser, to_nested_dicts
from liaison import utils as U
from liaison.daper import ConfigDict
from liaison.daper.milp.heuristics.integral_heuristic import \
    run as run_integral_heuristic
from liaison.daper.milp.heuristics.random_heuristic import \
    run as run_random_heuristic
from liaison.daper.milp.heuristics.spec import MILPHeuristic

parser = ArgumentParser()
parser.add_argument('--out_file', required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--n_local_moves', type=int, required=True)
parser.add_argument('--k', type=int, required=True)

# env_config
parser.add_config_file(name='env', required=True)

# random heuristic
parser.add_argument('--n_trials', type=int, required=True)
parser.add_argument('--random_seeds', type=int, nargs='+', required=True)
global args


def make_env():
  env_config = ConfigDict(to_nested_dicts(args.env_config))
  env_config.lp_features = False
  env_config.k = args.k
  env_config.steps_per_episode = args.k * args.n_local_moves
  env_config.primal_gap_reward = True
  env_config.delta_reward = False
  assert env_config.n_graphs == 1

  env_class = U.import_obj(env_config.class_name, env_config.class_path)
  env = env_class(id=0, seed=args.seed, **env_config)
  return env


def main(argv):
  global args
  args = parser.parse_args(argv[1:])
  heuristic = MILPHeuristic()

  res = run_random_heuristic(args.n_local_moves, args.n_trials,
                             args.random_seeds, make_env())
  heuristic.random.update(seeds=args.random_seeds,
                          n_local_moves=args.n_local_moves,
                          k=args.k,
                          results=res)

  res = run_integral_heuristic(True, args.n_local_moves, args.k, args.n_trials,
                               args.random_seeds, make_env())
  heuristic.least_integral.update(seeds=args.random_seeds,
                                  n_local_moves=args.n_local_moves,
                                  k=args.k,
                                  results=res)

  res = run_integral_heuristic(False, args.n_local_moves, args.k,
                               args.n_trials, args.random_seeds, make_env())
  heuristic.most_integral.update(seeds=args.random_seeds,
                                 n_local_moves=args.n_local_moves,
                                 k=args.k,
                                 results=res)

  path = Path(args.out_file)
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, 'wb') as f:
    pickle.dump(heuristic, f)


if __name__ == '__main__':
  app.run(main)
  # main()
