"""
python liaison/daper/milp/heuristics/run_dataset.py --dataset=milp-cauction-10 --n_training_samples=1 --n_valid_samples=1 --n_test_samples=1 --out_dir /tmp/heuristics --n_local_moves=10 --k=5
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from liaison.daper import ConfigDict
from liaison.daper.dataset_constants import (DATASET_PATH, LENGTH_MAP,
                                             NORMALIZATION_CONSTANTS)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument(
    '--n_training_samples',
    type=int,
    default=None,
    help=
    'If not specified, look up dataset size from liaison.daper.dataset_constants'
)
parser.add_argument('--n_valid_samples', type=int, default=None)
parser.add_argument('--n_test_samples', type=int, default=None)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--n_local_moves', type=int, required=True)
parser.add_argument('--k', type=int, required=True)
REMAINDER = ''

SEED = 42
N_TRIALS = 5


def preprocess(argv):
  if '--' in sys.argv:
    global REMAINDER
    idx = sys.argv.index('--')
    REMAINDER = ' '.join(sys.argv[idx + 1:])
  else:
    idx = len(sys.argv)
  return sys.argv[1:idx]


args = parser.parse_args(preprocess(sys.argv))


def cmd_gen(seed, out_file, graph_path, random_seeds, n_local_moves, k,
            dataset, dataset_type, graph_idx):
  cmd = f"""python {os.path.dirname(__file__)}/run_graph.py --
                    --seed={seed}
                    --out_file={out_file}
                    --random_seeds {' '.join(map(str, random_seeds))}
                    --n_trials={len(random_seeds)}
                    --n_local_moves={n_local_moves}
                    --env_config_file=liaison/configs/env/rins.py
                    --env_config.k={k}
                    --env_config.dataset={dataset}
                    --env_config.dataset_type={dataset_type}
                    --env_config.graph_start_idx={graph_idx}
                    --env_config.n_graphs=1
                    {REMAINDER}"""
  return ' '.join(cmd.replace('\n', ' ').split())


def main():
  cmds = []
  dataset_lengths = ConfigDict(LENGTH_MAP[args.dataset])

  if args.n_training_samples is not None:
    dataset_lengths.train = args.n_training_samples
  if args.n_valid_samples is not None:
    dataset_lengths.valid = args.n_valid_samples
  if args.n_test_samples is not None:
    dataset_lengths.test = args.n_test_samples

  random_seed = 0
  seed = SEED
  rng = np.random.RandomState(SEED)
  for mode in ['train', 'valid', 'test']:
    for i in rng.choice(range(LENGTH_MAP[args.dataset][mode]),
                        dataset_lengths[mode],
                        replace=False):
      out_file = Path(args.out_dir, mode, f'{i}.pkl')
      graph_path = Path(DATASET_PATH[args.dataset], mode, f'{i}.pkl')
      random_seeds = list(range(random_seed, random_seed + N_TRIALS))
      random_seed += N_TRIALS
      cmds += [
          cmd_gen(seed, out_file, graph_path, random_seeds, args.n_local_moves,
                  args.k, args.dataset, mode, i)
      ]
      seed += 1

  for cmd in cmds:
    print(cmd)


if __name__ == '__main__':
  main()