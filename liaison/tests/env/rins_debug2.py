# debug assert fails during training in rins
# by trying to find the cases where this happens using a uniform random
# agent against the rins environment.
import argparse
import os
import pdb
import pickle
import random
import sys
from multiprocessing.pool import Pool, ThreadPool

import liaison.utils as U
import numpy as np
import tree as nest
from absl import app
from liaison.agents.ur_discrete import Agent
from liaison.daper.dataset_constants import DATASET_PATH, LENGTH_MAP
from liaison.env.batch import ParallelBatchedEnv, SerialBatchedEnv
from liaison.env.rins import Env
from liaison.utils import ConfigDict
from pyscipopt import Model, multidict, quicksum

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--use_procs', action='store_true')
parser.add_argument('--k', type=int, default=20)
global args


def load_graphs():
  path = DATASET_PATH['milp-facilities-10']
  milps = []
  for graph_idx in range(10000):
    with open(os.path.join(path, 'train', f'{graph_idx}.pkl'), 'rb') as f:
      milp = pickle.load(f)
      milps.append(milp)
  return milps


def scip_solve(mip):
  """Solves a mip/lp using scip"""
  solver = Model()
  solver.hideOutput()
  mip.add_to_scip_solver(solver)
  solver.optimize()
  obj = float(solver.getObjVal())
  return obj


def unroll(milps):
  milp = np.random.choice(milps)
  opt_sol = milp.optimal_solution
  fixed_vars_to_vals = {
      var: opt_sol[var]
      for var in random.sample(list(opt_sol), args.k)
  }
  obj = scip_solve(milp.mip.unfix(fixed_vars_to_vals, integral_relax=False))
  assert obj >= milp.optimal_objective - 1e-4, (obj, milp.optimal_objective)


def run_debug():
  i = 0
  milps = load_graphs()
  N = 64
  pool = Pool(N)
  while True:
    if i % int(1e3) == 0:
      print('.', end='')
      sys.stdout.flush()

    pool.map(unroll, [milps] * N)

    i += 1


def main(argv):
  global args
  args = parser.parse_args(argv[1:])
  run_debug()


if __name__ == '__main__':
  app.run(main)
