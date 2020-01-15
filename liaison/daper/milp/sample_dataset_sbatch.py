"""
python liaison/daper/milp/sample_dataset.py --out_dir=/data/nms/tfp/datasets/milp/facilities/size-3/ --n_training_samples=100 --n_valid_samples=1 --n_test_samples=1 -- --problem_type=facilities --problem_size=3 | parallel --ungroup -j8
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--n_training_samples', type=int, default=1000)
parser.add_argument('--n_valid_samples', type=int, default=100)
parser.add_argument('--n_test_samples', type=int, default=100)
# turns on slurm mode.
parser.add_argument('--slurm_mode', action='store_true')
REMAINDER = ''


def write_sbatch_submit_script(n_cmds):
  s = f"""#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --nodes=cpu
#SBATCH --array=1-64

for i in {{1..{n_cmds}}}; do
  if (( $i % $SLURM_ARRAY_TASK_ID == 0 )); then
    srun $(head -n $i jobs.txt | tail -n 1)
    if [ $? -eq 0 ]; then true
    else
        >&2 echo FAIL
    fi
  fi
done
"""
  with open('sbatch.txt', 'w') as f:
    f.write(s)


def preprocess(argv):
  if '--' in sys.argv:
    global REMAINDER
    idx = sys.argv.index('--')
    REMAINDER = ' '.join(sys.argv[idx + 1:])
  else:
    idx = len(sys.argv)
  return sys.argv[1:idx]


args = parser.parse_args(preprocess(sys.argv))


def cmd_gen(seed, out_file):
  cmd = "python %s --seed=%d --out_file=%s %s" % (os.path.join(
      os.path.dirname(__file__), 'sample_graph.py'), seed, out_file, REMAINDER)
  return cmd


def main():
  seed = 0
  cmds = []
  for mode, size in zip(
      ['train', 'valid', 'test'],
      [args.n_training_samples, args.n_valid_samples, args.n_test_samples]):
    for i in range(size):
      out_file = os.path.join(args.out_dir, mode, '%d.pkl' % i)
      cmds += [cmd_gen(seed, out_file)]
      seed += 1

  for cmd in cmds:
    print(cmd)


if __name__ == '__main__':
  main()
