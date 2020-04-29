"""
python liaison/daper/milp/mine_dataset.py -- --out_dir=/data/nms/tfp/datasets/milp/facilities/size-3/ --problem_type=facilities --problem_size=3 | parallel --ungroup -j8
"""
import argparse
import math
import os
import shlex
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-N', '--n_sample_per_proc', type=int, default=1)
parser.add_argument('-j', '--n_procs', type=int, default=1)
parser.add_argument('--n_procs_per_srun', type=int, default=2)
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

  M = args.n_procs_per_srun
  for i in range(math.ceil(len(cmds) / float(M))):
    join_cmds = []
    for j in range(M):
      idx = M * i + j
      if idx < len(cmds):
        join_cmds.append(cmds[idx])
    cmd = ' & '.join(join_cmds)
    if len(join_cmds) > 1:
      cmd += '; wait'
    if args.slurm_mode:
      print(f'srun --overcommit --mem=2G {args.slurm_options} bash -c {shlex.quote(cmd)}')
    else:
      print(f'bash -c {shlex.quote(cmd)}')


if __name__ == '__main__':
  main()
