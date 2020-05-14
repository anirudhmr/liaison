"""
python liaison/daper/milp/heuristics/sweep.py --dataset=milp-cauction-10 --out_dir /tmp/heuristics --n_local_moves=10 --n_training_samples=1 --n_valid_samples=1 --n_test_samples=1
"""
import argparse
import json
import os
import pickle
import shlex
import sys

import liaison.utils as U
from liaison.daper import ConfigDict
from liaison.launch import hyper

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out_dir', type=str, required=True)
# turns on slurm mode.
parser.add_argument('-s', '--slurm_mode', action='store_true')


def main():
  args, remainder = parser.parse_known_args(sys.argv[1:])
  try:
    for work_id, params in enumerate(hyper.product(hyper.discrete('k', [15]), )):
      params = ConfigDict(params)
      d = '_'.join([f'{k}:{v}' for k, v in sorted(params.items())])
      cmd = f'python liaison/daper/milp/heuristics/run_dataset.py --out_dir={args.out_dir}/{d} --k={params.k} {" ".join(remainder)}'
      if args.slurm_mode:
        for cmd in U.run_cmd(cmd).split('\n'):
          cmd = cmd.rstrip('\n')
          if cmd:
            print(f'srun --mem=2G --overcommit bash -c "{cmd} > /dev/null"')
        print(
            f'mkdir -p {args.out_dir}; echo {shlex.quote(json.dumps(params))}  > "{args.out_dir}/{d}.json"'
        )
      else:
        os.system(cmd)
        print(
            f'mkdir -p {args.out_dir} && echo {shlex.quote(json.dumps(params))} > {args.out_dir}/{d}.json'
        )

  except KeyboardInterrupt:
    sys.exit(0)


if __name__ == '__main__':
  main()
