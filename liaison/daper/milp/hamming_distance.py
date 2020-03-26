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

from liaison.daper.dataset_constants import DATASET_PATH, LENGTH_MAP
from liaison.daper.milp.primitives import (BinaryVariable, Constraint,
                                           IntegerVariable, Objective)
from liaison.utils import ConfigDict
from pyscipopt import Model


def worker_fn(args, hamming_dist):
  with open(args.input_pkl_path, 'rb') as f:
    milp = pickle.load(f)

  constant = 0
  c = Constraint('LE', 0)
  for var in milp.mip.varname2var.values():
    if isinstance(var, BinaryVariable):
      if milp.optimal_solution[var.name] == 1:
        c.add_term(var.name, -1)
        constant += 1
      else:
        c.add_term(var.name, 1)

  # constant + c.expr = hamming_distance
  c.rhs = hamming_dist - constant
  c2 = copy.deepcopy(c)
  c2.sense = 'GE'

  mip = copy.deepcopy(milp.mip)
  mip.constraints.append(c)
  mip.constraints.append(c2)
  # remove the objective. Only feasibility is desired.
  mip.obj = Objective()

  solver = Model()
  solver.hideOutput()
  mip.add_to_scip_solver(solver)
  solver.optimize()

  if solver.getStatus() == 'infeasible':
    return None

  sol = {var.name: solver.getVal(var) for var in solver.getVars()}

  # get the objective for the above feasible solution
  def get_objective(sol):
    solver = Model()
    solver.hideOutput()

    milp.mip.add_to_scip_solver(solver)

    scip_sol = solver.createSol()
    for var in solver.getVars():
      solver.setSolVal(scip_sol, var, sol[var.name])
    return solver.getSolObjVal(scip_sol)

  return sol, get_objective(sol)


def determine_hamming_dists(args):
  # figure out the hamming distances to use
  with open(args.input_pkl_path, 'rb') as f:
    milp = pickle.load(f)

  # get hamming distance between the feasible solution and
  # the optimal solution
  hd = 0
  for var in milp.mip.varname2var.values():
    if isinstance(var, IntegerVariable):
      var_name = var.name
      hd += (milp.feasible_solution[var_name] != milp.optimal_solution[var_name])

  return list(range(1, (hd + 1)))


def spawner_fn(args):
  # generates slurm commands used to spawn processes to run.
  dataset_path = DATASET_PATH[args.dataset]
  cmds = []
  for dataset_type in ['train', 'valid', 'test']:
    for i in range(LENGTH_MAP[args.dataset][dataset_type]):
      inp_pkl_fname = f'{dataset_path}/{dataset_type}/{i}.pkl'
      out_pkl_fname = f'{args.out_path}/{dataset_type}/{i}.pkl'
      cmd = f'python {__file__} --worker_mode --input_pkl_path={inp_pkl_fname} --output_pkl_path={out_pkl_fname}'
      cmds += [cmd]

  for i in range(math.ceil(len(cmds) / 2)):
    if 2 * i + 1 < len(cmds):
      cmd = cmds[2 * i] + ' & ' + cmds[2 * i + 1] + '; wait'
    else:
      cmd = cmds[2 * i]
    print(f'srun --overcommit --mem=2G bash -c {shlex.quote(cmd)}')


def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('--worker_mode', action='store_true')
  args, _ = parser.parse_known_args(argv)
  worker_mode = args.worker_mode

  parser = argparse.ArgumentParser()
  parser.add_argument('--worker_mode', action='store_true')
  if worker_mode:
    parser.add_argument('--input_pkl_path', required=True)
    parser.add_argument('--output_pkl_path', required=True)
    args = parser.parse_args(argv)
    sols = dict()  # hamming_dist -> sol
    hamming_dists = determine_hamming_dists(args)
    for k in hamming_dists:
      ret = worker_fn(args, k)
      if ret is not None:
        sols[k] = ret
    Path(args.output_pkl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_pkl_path, 'wb') as f:
      pickle.dump(sols, f)
  else:
    parser.add_argument('--dataset', help='Must be registered in dataset_constants.py')
    parser.add_argument('--out_path', '--output_path', required=True, type=str)
    args = parser.parse_args(argv)
    spawner_fn(args)


if __name__ == '__main__':
  main(sys.argv[1:])
