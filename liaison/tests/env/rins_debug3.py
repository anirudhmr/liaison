# debug assert fails during training in rins
# by trying to find the cases where this happens using a uniform random
# agent against the rins environment.
import argparse
import os
import pdb
import pickle

import liaison.utils as U
import numpy as np
import tree as nest
from absl import app
from liaison.daper.dataset_constants import DATASET_PATH, LENGTH_MAP
from liaison.utils import ConfigDict
from pyscipopt import Model, multidict, quicksum

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--use_procs', action='store_true')
parser.add_argument('--k', type=int, default=20)
global args


def load_graph():
  path = DATASET_PATH['milp-facilities-10']
  with open(os.path.join(path, 'train', '4070.pkl'), 'rb') as f:
    return pickle.load(f)


def scip_solve(mip):
  """Solves a mip/lp using scip"""
  solver = Model()
  solver.hideOutput()
  mip.add_to_scip_solver(solver)
  solver.optimize()
  obj = float(solver.getObjVal())
  return obj


def unroll():
  milp = load_graph()
  with open('/home/ubuntu/43575.txt', 'rb') as f:
    pkl = pickle.load(f)

  ass = pkl['fixed_assignment']
  mip = milp.mip

  opt_sol = milp.optimal_solution
  obj = scip_solve(mip.unfix(ass, integral_relax=False))
  assert obj >= milp.optimal_objective - 1e-4, (obj, milp.optimal_objective)


def run_debug():
  unroll()


def main(argv):
  global args
  args = parser.parse_args(argv[1:])
  run_debug()


if __name__ == '__main__':
  app.run(main)
