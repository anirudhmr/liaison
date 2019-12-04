import argparse
import math
import os
import pickle
from typing import Dict, List, Tuple

from absl import app
from liaison.daper.dataset_constants import DATASET_PATH, LENGTH_MAP
from liaison.daper.milp.dataset import MILP
from pyscipopt import Model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='milp-facilities-10')
parser.add_argument('--dataset_type', default='train')
parser.add_argument('--graph_idx', type=int, default=0)
args = None


def scip_solve(mip):
  """Solves a mip/lp using scip"""
  solver = Model()
  solver.hideOutput()
  mip.add_to_scip_solver(solver)
  solver.optimize()
  obj = solver.getObjVal()
  return obj


def testRelax(milp):
  feasible_sol = milp.feasible_solution
  optimal_sol = milp.optimal_solution

  fix_variables = dict()
  for k in feasible_sol.keys():
    # thresholding
    if math.fabs(feasible_sol[k] - optimal_sol[k]) <= 1e-4:
      fix_variables[k] = feasible_sol[k]
  print(len(fix_variables))
  obj = scip_solve(milp.mip.unfix(fix_variables, integral_relax=False))
  assert obj == milp.optimal_objective


def compute_feas_to_opt(milp):

  def f(d: Dict[str, int]):
    # extract sorted values
    return [v for _, v in sorted([(k, v) for k, v in d.items()])]

  feasible_sol = f(milp.feasible_solution)
  optimal_sol = f(milp.optimal_solution)

  n_matches = 0
  for x, y in zip(feasible_sol, optimal_sol):
    # thresholding
    if math.fabs(x - y) <= 1e-4:
      n_matches += 1

  print('# matches between feasible and optimal sol; # differences')
  print(n_matches, len(feasible_sol) - n_matches)
  print('feasible_obj, feasible_sol:')
  print(milp.feasible_objective, feasible_sol)
  print('optimal_obj, optimal_sol:')
  print(milp.optimal_objective, optimal_sol)
  return n_matches


def load_graph(dataset: str, dataset_type: str, graph_idx: int):
  path = DATASET_PATH[dataset]
  assert graph_idx < LENGTH_MAP[dataset][dataset_type]
  with open(os.path.join(path, dataset_type, '%d.pkl' % graph_idx), 'rb') as f:
    milp = pickle.load(f)
  return milp


def main(argv):
  global args
  args = parser.parse_args(argv[1:])
  milp = load_graph(args.dataset, args.dataset_type, args.graph_idx)
  testRelax(milp)
  compute_feas_to_opt(milp)


if __name__ == '__main__':
  app.run(main)
