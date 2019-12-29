"""
python liaison/daper/milp/heuristics/sweep.py --dataset=milp-cauction-10 --out_dir /tmp/heuristics --n_local_moves=10 --k=5 --n_training_samples=1 --n_valid_samples=1 --n_test_samples=1
"""
import argparse
import json
import os
import pickle
import shlex
import sys

from liaison.daper import ConfigDict
from liaison.launch import hyper

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, required=True)
# turns on slurm mode.
parser.add_argument('--slurm_mode', action='store_true')


def write_sbatch_submit_script(n_cmds):
  s = f"""#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --array=1-{min(256, n_cmds)}

for i in {{1..{n_cmds}}}; do
  if (( $i % $SLURM_ARRAY_TASK_ID == 0 )) && (( $i <= {n_cmds} )); then
    srun $(head -n $i jobs.txt | tail -n 1) 2> /dev/null
    if [ $? -eq 0 ]; then true
    else
        >&2 echo FAIL
    fi
  fi
done
"""
  with open('sbatch.txt', 'w') as f:
    f.write(s)


def main():
  args, remainder = parser.parse_known_args(sys.argv[1:])
  os.system('echo -n "" > jobs.txt')
  try:
    for work_id, params in enumerate(
        hyper.product(hyper.discrete('k', [1, 2, 3, 5, 10, 15, 20]), )):
      params = ConfigDict(params)
      d = '_'.join([f'{k}:{v}' for k, v in sorted(params.items())])
      cmd = f'python liaison/daper/milp/heuristics/run_dataset.py --out_dir={args.out_dir}/{d} --k={params.k} {" ".join(remainder)}'
      if args.slurm_mode:
        os.system(f'{cmd} >> jobs.txt')
        with open('jobs.txt', 'a') as f:
          print(
              f'srun -o {args.out_dir}/{d}.json echo "{shlex.quote(json.dumps(params))}"',
              file=f)
      else:
        os.system(cmd)
        print(
            f'echo "{shlex.quote(json.dumps(params))}" > {args.out_dir}/{d}.json'
        )

    if args.slurm_mode:
      write_sbatch_submit_script(sum(1 for line in open('jobs.txt')))

  except KeyboardInterrupt:
    sys.exit(0)


if __name__ == '__main__':
  main()
