# Calculate feasible solutions which are at
# specified hamming distances away from the optimal

import argparse
import copy
import math
import pickle
import shlex
import sys
from pathlib import Path
from typing import Dict, List

from absl import app
from liaison.daper.dataset_constants import DATASET_PATH, LENGTH_MAP
from liaison.daper.milp.dataset import AUXILIARY_MILP, MILP
from liaison.daper.milp.features import (get_features_from_scip_model,
                                         init_scip_params)
from liaison.daper.milp.primitives import (BinaryVariable, Constraint,
                                           IntegerVariable, Objective,
                                           relax_integral_constraints)
from liaison.utils import ConfigDict
from pyscipopt import Model


def worker_fn(args):
  with open(args.input_pkl_path, 'rb') as f:
    milp = pickle.load(f)
  aux = AUXILIARY_MILP()

  def find_optimal_sol(milp):
    solver = Model()
    solver.hideOutput()
    relax_integral_constraints(milp.mip).add_to_scip_solver(solver)
    solver.optimize()
    assert solver.getStatus() == 'optimal', solver.getStatus()
    ass = {var.name: solver.getVal(var) for var in solver.getVars()}
    return ass

  aux.optimal_lp_sol = find_optimal_sol(milp)
  solver = Model()
  solver.hideOutput()
  milp.mip.add_to_scip_solver(solver)
  aux.mip_features = get_features_from_scip_model(solver)
  return aux


def spawner_fn(args):
  # generates slurm commands used to spawn processes to run.
  dataset_path = DATASET_PATH[args.dataset]
  cmds = []
  for dataset_type in ['train', 'valid', 'test']:
    for i in range(LENGTH_MAP[args.dataset][dataset_type]):
      inp_pkl_fname = f'{dataset_path}/{dataset_type}/{i}.pkl'
      out_pkl_fname = f'{args.out_path}/{dataset_type}/{i}.pkl'
      cmds += [
          f'python {__file__} -- --worker_mode --input_pkl_path={inp_pkl_fname} --output_pkl_path={out_pkl_fname}'
      ]

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


def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--worker_mode', action='store_true')
  args, _ = parser.parse_known_args(argv[1:])
  worker_mode = args.worker_mode

  parser = argparse.ArgumentParser()
  parser.add_argument('--worker_mode', action='store_true')
  if worker_mode:
    parser.add_argument('--input_pkl_path', required=True)
    parser.add_argument('--output_pkl_path', required=True)
    args = parser.parse_args(argv[1:])

    Path(args.output_pkl_path).parent.mkdir(parents=True, exist_ok=True)
    try:
      p = worker_fn(args)
      with open(args.output_pkl_path, 'wb') as f:
        pickle.dump(p, f)
    except AssertionError:
      pass
  else:
    parser.add_argument('-d', '--dataset', help='Must be registered in dataset_constants.py')
    parser.add_argument('-o', '--out_path', '--output_path', required=True, type=str)
    parser.add_argument('--n_procs_per_srun', type=int, default=2)
    parser.add_argument('-s', '--slurm_mode', action='store_true')
    args = parser.parse_args(argv[1:])
    spawner_fn(args)


if __name__ == '__main__':
  app.run(main)
