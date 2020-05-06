import argparse
import math
import os
import shlex
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inp_dir', required=True)
parser.add_argument('-o', '--out_dir', required=True)
parser.add_argument('-s', '--slurm_mode', action='store_true')
parser.add_argument('--n_procs_per_srun', type=int, default=2)
args, REMAINDER = parser.parse_known_args()


def cmd_gen(inp_file, out_file, problem_type):
  cmd = "python %s/convert_graph.py --inp_file=%s --out_file=%s --problem_type=%s %s" % (
      os.path.dirname(__file__), inp_file, out_file, problem_type, ' '.join(REMAINDER))
  return cmd


def main():
  cmds = []
  for i, fname in enumerate(os.listdir(args.inp_dir)):
    out_file = os.path.join(args.out_dir, f'{i}.pkl')
    full_fname = Path(args.inp_dir) / fname
    cmds += [cmd_gen(full_fname, out_file, full_fname)]

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
      print(f'srun --overcommit --mem=2G bash -c {shlex.quote(cmd)}')
    else:
      print(f'bash -c {shlex.quote(cmd)}')


if __name__ == '__main__':
  main()
