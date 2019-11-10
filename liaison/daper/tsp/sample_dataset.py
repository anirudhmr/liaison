"""
python tsp/sample_dataset.py --out_dir=/tmp/concorde --num_nodes=100 --n_training_samples=1
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--num_nodes', type=int, required=True)
parser.add_argument('--n_training_samples', type=int, default=1000)
parser.add_argument('--n_valid_samples', type=int, default=0)
parser.add_argument('--n_test_samples', type=int, default=0)
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


def cmd_gen(seed, out_file):
  envs = 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/scratch/arc/gurobi811/linux64/lib/ '
  cmd = "bash -c 'python %s --seed=%d --num_nodes=%d --out_file=%s'" % (
      os.path.join(os.path.dirname(__file__),
                   'sample_graph.py'), seed, args.num_nodes, out_file)
  return envs + cmd


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
