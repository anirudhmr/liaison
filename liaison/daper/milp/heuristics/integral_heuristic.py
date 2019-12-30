from math import fabs

import numpy as np

from liaison.daper import ConfigDict
from liaison.daper.milp.primitives import IntegerVariable
from liaison.env import StepType
from liaison.env.rins import Env
from pyscipopt import Model


def scip_solve(mip):
  """Solves a mip/lp using scip"""
  solver = Model()
  solver.hideOutput()
  mip.add_to_scip_solver(solver)
  solver.optimize()
  assert solver.getStatus() == 'optimal', solver.getStatus()
  ass = {var.name: solver.getVal(var) for var in solver.getVars()}
  return ass


def choose_act(curr_sol, mip, rng, k, least_integral=True):
  errs = []
  for var_name, var in mip.varname2var.items():
    if isinstance(var, IntegerVariable):
      partial_sol = curr_sol.copy()
      # unfix current variable
      del partial_sol[var_name]
      submip = mip.fix(partial_sol, relax_integral_constraints=False)
      ass = scip_solve(submip)
      err = fabs(ass[var_name] - curr_sol[var_name])
      errs.append((err, var_name))

  if least_integral:
    errs = sorted(errs, reverse=True)
  else:
    errs = sorted(errs)

  var_names = [var for err, var in errs[0:k]]
  for err, var_name in errs:
    if err == errs[k - 1][0]:
      if var_name not in var_names:
        var_names.append(var_name)
  return rng.choice(var_names, size=k, replace=False)


def run(least_integral, n_local_moves, k, n_trials, seeds, env):
  assert len(seeds) == n_trials

  log_vals = [[] for _ in range(n_trials)]
  for trial_i, seed in zip(range(n_trials), seeds):
    rng = np.random.RandomState(seed)
    ts = env.reset()
    obs = ConfigDict(ts.observation)
    log_vals[trial_i].append(obs.log_values)

    while obs.graph_features.globals[Env.GLOBAL_N_LOCAL_MOVES] < n_local_moves:
      var_names = choose_act(env._curr_soln, env.milp.mip, rng, k,
                             least_integral)
      for var_name in var_names:
        act = env._var_names.index(var_name)
        ts = env.step(act)

      obs = ConfigDict(ts.observation)
      assert obs.graph_features.globals[Env.GLOBAL_LOCAL_SEARCH_STEP]

      log_vals[trial_i].append(obs.log_values)

    assert ts.step_type == StepType.LAST
  return log_vals
