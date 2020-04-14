import pickle

import liaison.utils as U
import numpy as np
from absl import app
from liaison.daper.milp.scip_mip import SCIPMIPInstance
from liaison.env.utils.rins import get_sample
from tqdm import tqdm

EPSILON = 1e-3


def scip_solve(solver):
  """solves a mip/lp using scip"""
  # solver.hideOutput()
  solver.setBoolParam('randomization/permutevars', True)
  # seed is set to 0 permanently.
  solver.setIntParam('randomization/permutationseed', 0)
  solver.setIntParam('randomization/randomseedshift', 0)

  solver.optimize()
  assert solver.getStatus() == 'optimal', solver.getStatus()
  obj = float(solver.getObjVal())
  ass = {var.name: solver.getVal(var) for var in solver.getVars()}
  return ass, obj


def fix_and_solve(model, fixed_ass):
  # fix the upper and lower bounds for the variables.

  fixed_model_vars = list(model.getVars(transformed=True))
  fixed_model_varname2var = {v.name.lstrip('t_'): v for v in fixed_model_vars}

  # restore to the original -- needed when reusing submip model.
  for v, var in fixed_model_varname2var.items():
    l, u = var.getLbOriginal(), var.getUbOriginal()
    model.chgVarLbGlobal(var, l - EPSILON)
    model.chgVarUbGlobal(var, u + EPSILON)

  for v, val in fixed_ass.items():
    var = fixed_model_varname2var[v]
    model.chgVarLbGlobal(var, val - EPSILON)
    model.chgVarUbGlobal(var, val + EPSILON)
  return scip_solve(model)


def main(_):
  milp = get_sample('milp-cauction-300-filtered', 'train', 102)
  mip = SCIPMIPInstance.fromMIPInstance(milp.mip)
  times = []
  for _ in range(10):
    with U.Timer() as timer:
      model = mip.get_scip_model()
    times.append(timer.to_seconds())

  print(f'Avg time to copy the model: {np.mean(times[5:])}')

  for i in tqdm(range(20)):
    fixed_ass = {
        k: milp.feasible_solution[k]
        for k in np.random.permutation(list(milp.feasible_solution.keys()))[:500]
    }
    ass, obj = fix_and_solve(model, fixed_ass)
    print(obj)
    model.freeTransform()


if __name__ == '__main__':
  app.run(main)
