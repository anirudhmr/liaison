"""
python liaison/daper/milp/mine_dataset.py --out_dir=/data/nms/tfp/datasets/milp/facilities/size-3/ -- --problem_type=facilities --problem_size=3 | parallel --ungroup -j8
"""
import argparse
import math
import os
import shlex
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-N', '--n_sample_per_proc', type=int, default=1)
parser.add_argument('-j', '--n_procs', type=int, default=1)
parser.add_argument('-s', '--slurm_mode', action='store_true')
REMAINDER = ''


def preprocess(argv):
  if '--' in sys.argv:
    global REMAINDER
    idx = sys.argv.index('--')
    REMAINDER = ' '.join(sys.argv[idx + 1:])
  else:
    idx = len(sys.argv)
  return sys.argv[1:idx]


args = parser.parse_args(preprocess(sys.argv))


def cmd_gen(seed):
  script_fname = os.path.join(os.path.dirname(__file__), 'mine_graph.py')
  cmd = f"python {script_fname} --seed={seed} --n_samples={args.n_sample_per_proc} {REMAINDER}"
  return cmd


def main():
  seed = 0
  cmds = []
  for i in range(args.n_procs):
    cmds.append(cmd_gen(seed))
    seed += args.n_sample_per_proc

  if args.slurm_mode:
    for i in range(math.ceil(len(cmds) / 2)):
      if 2 * i + 1 < len(cmds):
        cmd = cmds[2 * i] + ' & ' + cmds[2 * i + 1] + '; wait'
      else:
        cmd = cmds[2 * i]
      print(f'srun --overcommit --mem=2G bash -c {shlex.quote(cmd)}')
  else:
    for cmd in cmds:
      print(cmd)


if __name__ == '__main__':
  main()
