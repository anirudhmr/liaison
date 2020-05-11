import functools
import math
import pdb
from math import fabs

import numpy as np
from liaison.daper import ConfigDict
from liaison.daper.milp.primitives import (IntegerVariable,
                                           relax_integral_constraints)
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


def integral(curr_sol, mip, rng, k, *args, least_integral=True):
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


def rins(curr_sol, mip, rng, k, *args):
  errs = []
  for var, val in curr_sol.items():
    if isinstance(mip.varname2var[var], IntegerVariable):
      err = fabs(val - continuous_sol[var])
      errs.append((err, var))
  errs = sorted(errs, reverse=True)
  var_names = [var for err, var in errs[0:k]]
  for err, var_name in errs:
    if err == errs[k - 1][0]:
      if var_name not in var_names:
        var_names.append(var_name)
  # unfix variables.
  return rng.choice(var_names, size=k, replace=False)


def rens(curr_sol, mip, rng, k, env, *args):
  # return k variables to unfix.
  assert env.config.use_rens_submip_bounds
  errs = []
  optimal_lp_sol = mip.optimal_lp_sol
  for var, val in curr_sol.items():
    if isinstance(mip.varname2var[var], IntegerVariable):
      # if lp relaxation is not integral -- consider it to unfix  .
      if fabs(math.modf(optimal_lp_sol[var])[0]) <= 1e-3:
        errs.append((fabs(val - optimal_lp_sol[var]), var))
  errs = sorted(errs, reverse=True)
  var_names = [var for err, var in errs[0:k]]
  for err, var_name in errs:
    if err == errs[k - 1][0]:
      if var_name not in var_names:
        var_names.append(var_name)
  return rng.choice(var_names, size=k, replace=False)


def random(curr_sol, mip, rng, k, env):
  mask = env._variable_nodes[:, Env.VARIABLE_IS_INTEGER_FIELD]
  assert np.sum(mask) > 0
  indices = rng.choice(len(mask), size=k, replace=False, p=np.float32(mask) / np.sum(mask))
  return [env._var_names[i] for i in indices]


CHOOSE_HEURISTIC = dict(
    random=random,
    rins=rins,
    rens=rens,
    least_integral=functools.partial(integral, least_integral=True),
    most_integral=functools.partial(integral, least_integral=False),
)


def run(heuristic, k, n_trials, seeds, env, muldi_actions=False):
  assert len(seeds) == n_trials

  heuristic_fn = CHOOSE_HEURISTIC[heuristic]
  log_vals = [[] for _ in range(n_trials)]
  for trial_i, seed in zip(range(n_trials), seeds):
    rng = np.random.RandomState(seed)
    ts = env.reset()
    obs = ConfigDict(ts.observation)
    log_vals[trial_i].append(dict(obs.curr_episode_log_values))

    while ts.step_type != StepType.LAST:
      var_names = heuristic_fn(env.get_curr_soln(), env.milp.mip, rng, k, env)
      if muldi_actions:
        act = [env._var_names.index(var_name) for var_name in var_names]
        ts = env.step(act)
      else:
        for i, var_name in enumerate(var_names):
          act = env._var_names.index(var_name)
          ts = env.step(act)
          if i != len(var_names) - 1:
            assert ts.step_type != StepType.LAST
      obs = ConfigDict(ts.observation)
      assert obs.graph_features.globals[Env.GLOBAL_LOCAL_SEARCH_STEP]
      log_vals[trial_i].append(dict(obs.curr_episode_log_values))
  return log_vals
