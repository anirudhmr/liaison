import argparse
import functools
import os
import pickle
from math import fabs
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
from absl import app
from liaison.daper import ConfigDict
from liaison.daper.dataset_constants import DATASET_PATH
from liaison.daper.milp.primitives import (BinaryVariable, ContinuousVariable,
                                           IntegerVariable,
                                           relax_integral_constraints)
from liaison.env import StepType
from pyscipopt import Model

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset')
parser.add_argument('-T', '--dataset_type', default='train')
parser.add_argument('-N', '--n_samples', default=1, type=int)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_steps', type=int, default=int(1e6))
parser.add_argument('--out_dir', required=True)
args = None


def get_sample(dataset, dataset_type, graph_idx):
  dataset_path = DATASET_PATH[dataset]
  with open(os.path.join(dataset_path, dataset_type, f'{graph_idx}.pkl'),
            'rb') as f:
    milp = pickle.load(f)
  return milp


def greedy(curr_sol, mip, rng, n_steps, optimal_obj):
  # Applicable only for binary MIPS
  # Randomly selects a binary variable and flips its current assignment value if
  # it improves the objective without violating any constraint.
  # If a constraint is violated, restart the episode.
  # Report the best solution found.
  # validate that the MIP is binary.
  # Currently does not handle continuous variables.

  for var in mip.varname2var.values():
    if isinstance(var, ContinuousVariable):
      raise Exception(
          'Continuous varaibles should not be present as they are not currently handled correctly.'
      )
    assert isinstance(var, BinaryVariable)

  sol = curr_sol
  best_obj = mip.obj.expr.reduce(sol).constant
  raise Exception('Heuristic incomplete!')
  # result is list of tuples where the best solution has been updated
  result = [(0, best_obj / optimal_obj)]
  for step in range(n_steps):
    if step % int(1e4) == 0:
      print('.', end='', flush=True)

    def f(var, sol, best_obj):
      # get improvement in obj on flipping var v
      # returns None if any constraint is violated
      sol2 = dict(**sol)
      sol2.update({var: 1 - sol[var]})

      # If objective improves
      if mip.obj.expr.reduce(sol2).constant < best_obj:
        violates_constraint = False
        for c in mip.constraints:
          try:
            assert c.validate_assignment(sol2)
          except AssertionError:
            violates_constraint = True
            break
        if not violates_constraint:
          sol = sol2
          best_obj = mip.obj.expr.reduce(sol2).constant
          return best_obj, sol
      return None

    var = np.random.choice(list(sol.keys()), 1)[0]
    sol2 = dict(**sol)
    sol2.update({var: 1 - sol[var]})

    # If objective improves
    if mip.obj.expr.reduce(sol2).constant < best_obj:
      violates_constraint = False
      for c in mip.constraints:
        try:
          assert c.validate_assignment(sol2)
        except AssertionError:
          violates_constraint = True
          break
      if not violates_constraint:
        sol = sol2
        best_obj = mip.obj.expr.reduce(sol2).constant
        result.append((step, best_obj / optimal_obj))
  return result


def f(i):
  rng = np.random.RandomState(args.seed + i)
  milp = get_sample(args.dataset, args.dataset_type, i)
  res = greedy(milp.feasible_solution, milp.mip, rng, args.n_steps,
               milp.optimal_objective)
  p = Path(args.out_dir) / Path(f'{i}.pkl')
  p.parent.mkdir(parents=True, exist_ok=True)
  with open(p, 'wb') as f:
    pickle.dump(res, f)


def main(argv):
  global args
  args = parser.parse_args(argv[1:])

  with Pool() as pool:
    pool.map(f, range(args.n_samples))


if __name__ == '__main__':
  app.run(main)
