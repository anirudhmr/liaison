# Converts graph from mps or other standard formats to custom formatimport argparse
import argparse
import functools
import os
import pickle
from multiprocessing.pool import ThreadPool

import numpy as np
from liaison.daper.milp.dataset import MILP
from liaison.daper.milp.generate_graph import generate_instance
from liaison.daper.milp.primitives import scip_to_milps
from pyscipopt import (SCIP_HEURTIMING, SCIP_PARAMSETTING, SCIP_RESULT, Heur,
                       Model)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inp_file', type=str, required=True)
parser.add_argument('-o', '--out_file', type=str, required=True)
parser.add_argument('-t', '--time_limit', type=int, default=None)
parser.add_argument('--gap', type=float, default=0.)
parser.add_argument('--problem_type', type=str, required=True)
args = parser.parse_args()


class LogBestSol(Heur):

  def __init__(self):
    super(LogBestSol, self).__init__()
    self.primal_integral = 0.
    self.i = 0
    # list of tuples of (primal gap switch step, primal gap)
    self.l = []

  def heurexec(self, heurtiming, nodeinfeasible):
    sol = self.model.getBestSol()
    obj = self.model.getSolObjVal(sol)
    self.primal_integral += obj
    if self.l:
      if self.l[-1][1] != obj:
        self.l.append((self.i, obj))
    else:
      self.l.append((self.i, obj))
    self.i += 1
    return dict(result=SCIP_RESULT.DELAYED)


def convert():
  milp = MILP()
  milp.problem_type = args.problem_type

  model = Model()
  model.hideOutput()
  heur = LogBestSol()
  model.includeHeur(heur,
                    "PyHeur",
                    "custom heuristic implemented in python",
                    "Y",
                    timingmask=SCIP_HEURTIMING.BEFORENODE)

  model.setRealParam('limits/gap', args.gap)
  model.readProblem(args.inp_file)
  milp.mip = scip_to_milps(model)

  model.optimize()
  milp.optimal_objective = model.getObjVal()
  milp.optimal_solution = {
      var.name: model.getVal(var)
      for var in model.getVars()
  }
  milp.is_optimal = (model.getStatus() == 'optimal')
  milp.optimal_sol_metadata.n_nodes = model.getNNodes()
  milp.optimal_sol_metadata.gap = model.getGap()
  milp.optimal_sol_metadata.primal_integral = heur.primal_integral
  milp.optimal_sol_metadata.primal_gaps = heur.l

  feasible_sol = model.getSols()[-1]
  milp.feasible_objective = model.getSolObjVal(feasible_sol)
  milp.feasible_solution = {
      var.name: feasible_sol[var]
      for var in model.getVars()
  }
  return milp


def main():

  milp = convert()
  os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
  with open(args.out_file, 'wb') as f:
    pickle.dump(milp, f)


if __name__ == '__main__':
  main()
