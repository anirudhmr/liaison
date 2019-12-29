"""
python liaison/daper/milp/heuristics/sweep.py --dataset=milp-cauction-10 --out_dir /tmp/heuristics --n_local_moves=10 --k=5 --n_training_samples=1 --n_valid_samples=1 --n_test_samples=1
"""
import argparse
import os
import pickle
import sys

from absl import app
from liaison.daper import ConfigDict
from liaison.launch import hyper

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, required=True)


def main():
  args, remainder = parser.parse_known_args(sys.argv[1:])
  try:
    for work_id, params in enumerate(
        hyper.product(hyper.discrete('k', [1, 2, 3, 5, 10, 15, 20]),
                      hyper.discrete('n_local_moves', [1, 2, 5, 10, 20, 50]))):
      params = ConfigDict(params)
      d = '_'.join([f'{k}:{v}' for k, v in sorted(params.items())])
      os.system(
          f'python liaison/daper/milp/heuristics/run_dataset.py --out_dir={args.out_dir}/{d} --k={params.k} --n_local_moves={params.n_local_moves} {" ".join(remainder)}'
      )
      print(f'echo "{params}" > {args.out_dir}/{d}.json')
      sys.exit(0)
  except KeyboardInterrupt:
    sys.exit(0)


if __name__ == '__main__':
  main()
