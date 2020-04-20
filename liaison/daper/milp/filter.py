import argparse
import os
import pdb
import pickle
import shlex
import sys
from math import ceil, fabs
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

import numpy as np
from absl import app
from liaison.daper import ConfigDict
from liaison.daper.dataset_constants import (DATASET_PATH, LENGTH_MAP,
                                             NORMALIZATION_CONSTANTS)

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--slurm_mode', action='store_true')
parser.add_argument('-i', '--input_dir', required=True)
parser.add_argument('-N', '--n_read_samples', type=int, default=None)
parser.add_argument('-o', '--output_dir', required=True)
parser.add_argument('-Tr', '--n_training_samples', type=int, default=None)
parser.add_argument('-V', '--n_valid_samples', type=int, default=None)
parser.add_argument('-Te', '--n_test_samples', type=int, default=None)
parser.add_argument('-dr', '--dont_regenerate', action='store_true')
REMAINDER = ''
args = None


def read_pkl(fname):
  with open(fname, 'rb') as f:
    try:
      return pickle.load(f)
    except EOFError:
      pass


def get(d, k):
  for i in k.split('/'):
    d = d[i]
  return d


def primal_gap(curr_obj, optimal_obj):
  if curr_obj == 0 and optimal_obj == 0:
    rew = 0.
  elif np.sign(curr_obj) * np.sign(optimal_obj) < 0:
    rew = 1.
  else:
    rew = fabs(optimal_obj - curr_obj) / max(fabs(optimal_obj), fabs(curr_obj))
  return rew


def compute_primal_integral(sample):
  n_nodes = get(sample, 'optimal_sol_metadata/n_nodes')
  if n_nodes == 0:
    return None
  optimal_objective = get(sample, 'optimal_objective')
  l = list(get(sample, 'optimal_sol_metadata/primal_gaps'))
  l += [(n_nodes, optimal_objective)]
  integral = 0.
  for idx, (i, obj) in enumerate(l[:-1]):
    gap = primal_gap(obj, optimal_objective)
    integral += (gap * (l[idx + 1][0] - i))
  return integral


def compute_score(fname):
  sample = read_pkl(fname)
  if sample:
    score = compute_primal_integral(sample)
  else:
    score = None
  return (score, fname)


def _filter():
  scores = []
  if args.n_read_samples is None:
    files = [
        f'{args.input_dir}/{fname}' for fname in os.listdir(args.input_dir)
        if fname.endswith('.pkl')
    ]
  else:
    files = [f'{args.input_dir}/{i}.pkl' for i in range(args.n_read_samples)]

  with ThreadPool() as pool:
    scores = pool.map(compute_score, files)
  return list(filter(lambda k: k[0], scores))


def shuffle(dataset):
  rng = np.random.RandomState(0)
  return rng.permutation(dataset)


def split(dataset):
  return np.split(dataset, [
      args.n_training_samples, args.n_training_samples + args.n_valid_samples,
      args.n_training_samples + args.n_valid_samples + args.n_test_samples
  ])


def cmd_gen(seed, out_file, problem_size):
  cmd = "python %s --seed=%d --out_file=%s --problem_size=%d %s" % (os.path.join(
      os.path.dirname(__file__),
      'sample_graph.py'), seed, out_file, problem_size, ' '.join(REMAINDER))
  return cmd


def generate_cmds(train, valid, test):

  def f(mode, dataset, cmds):
    for i, sample in enumerate(dataset):
      sample = read_pkl(Path(args.input_dir) / sample[1])
      if sample is None:
        continue
      out_file = Path(args.output_dir) / mode / f'{i}.pkl'
      # problem_size added in new version.
      cmds += [cmd_gen(sample.seed, out_file, sample['problem_size'])]

  cmds = []
  f('train', train, cmds)
  f('valid', valid, cmds)
  f('test', test, cmds)
  return cmds


def main(argv):
  global args, REMAINDER
  args, REMAINDER = parser.parse_known_args(argv[1:])

  dataset = _filter()
  dataset = sorted(dataset, reverse=True)
  dataset = dataset[:args.n_training_samples + args.n_valid_samples + args.n_test_samples]
  dataset = shuffle(dataset)
  l = split(dataset)
  train = l[0]
  valid = l[1]
  test = l[2]
  if args.dont_regenerate:
    # In this case, the mined file is simply copied to the dataset.
    # slurm mode not applicable.
    def f(mode, dataset):
      for i, sample in enumerate(dataset):
        print(
            f'mkdir -p {args.output_dir}/{mode} && cp {sample[1]} {args.output_dir}/{mode}/{i}.pkl'
        )

    f('train', train)
    f('valid', valid)
    f('test', test)
  else:
    cmds = generate_cmds(train, valid, test)

    if args.slurm_mode:
      for i in range(ceil(len(cmds) / 2)):
        if 2 * i + 1 < len(cmds):
          cmd = cmds[2 * i] + ' & ' + cmds[2 * i + 1] + '; wait'
        else:
          cmd = cmds[2 * i]
        print(f'srun --overcommit --mem=2G bash -c {shlex.quote(cmd)}')
    else:
      for cmd in cmds:
        print(cmd)


if __name__ == '__main__':
  app.run(main)
