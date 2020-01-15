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
args, REMAINDER = parser.parse_known_args()


def cmd_gen(inp_file, out_file, problem_type):
  cmd = "python %s/convert_graph.py --inp_file=%s --out_file=%s --problem_type=%s %s" % (
      os.path.dirname(__file__), inp_file, out_file, problem_type,
      ' '.join(REMAINDER))
  return cmd


def main():
  cmds = []
  for i, fname in enumerate(os.listdir(args.inp_dir)):
    out_file = os.path.join(args.out_dir, f'{i}.pkl')
    full_fname = Path(args.inp_dir) / fname
    cmds += [cmd_gen(full_fname, out_file, full_fname)]

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
