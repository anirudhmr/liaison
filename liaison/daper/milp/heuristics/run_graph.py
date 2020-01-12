import os
import pickle
import sys
from pathlib import Path

import tree as nest
from absl import app
from argon import ArgumentParser, to_nested_dicts
from liaison import utils as U
from liaison.daper import ConfigDict
from liaison.daper.milp.heuristics.heuristic_fn import run as run_heuristic
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
parser.add_argument('--run_random_only', action='store_true')
parser.add_argument('--disable_maxcuts', action='store_true')
global args


def make_env():
  env_config = ConfigDict(to_nested_dicts(args.env_config))
  env_config.lp_features = False
  env_config.k = args.k
  env_config.n_local_moves = args.n_local_moves
  env_config.primal_gap_reward = True
  env_config.delta_reward = False
  env_config.disable_maxcuts = args.disable_maxcuts
  assert env_config.n_graphs == True

  env_class = U.import_obj(env_config.class_name, env_config.class_path)
  env = env_class(id=0, seed=args.seed, **env_config)
  return env


def main(argv):
  global args
  args = parser.parse_args(argv[1:])
  heuristic = MILPHeuristic()

  for k in heuristic.keys():
    for key in heuristic[k].keys():
      heuristic[k][key] = None

  res = run_random_heuristic(args.n_trials, args.random_seeds, make_env())
  heuristic.random.update(seeds=args.random_seeds,
                          n_local_moves=args.n_local_moves,
                          k=args.k,
                          results=res)

  if not args.run_random_only:
    # integral heuristics too slow...
    # disable temporarily
    # res = run_heuristic('least_integral', args.k, args.n_trials,
    #                     args.random_seeds, make_env())
    # heuristic.least_integral.update(seeds=args.random_seeds,
    #                                 n_local_moves=args.n_local_moves,
    #                                 k=args.k,
    #                                 results=res)

    # res = run_heuristic('most_integral', args.k, args.n_trials,
    #                     args.random_seeds, make_env())
    # heuristic.most_integral.update(seeds=args.random_seeds,
    #                                n_local_moves=args.n_local_moves,
    #                                k=args.k,
    #                                results=res)

    res = run_heuristic('rins', args.k, args.n_trials, args.random_seeds,
                        make_env())
    heuristic.rins.update(seeds=args.random_seeds,
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
