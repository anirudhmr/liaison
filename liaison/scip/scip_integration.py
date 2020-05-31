# Integrate scip with external RL environment
from math import fabs

import liaison.utils as U
from liaison.utils import ConfigDict
from pyscipopt import SCIP_HEURTIMING, SCIP_PARAMSETTING, SCIP_RESULT, Heur


class EvalHeur(Heur):

  def __init__(self, improve_sol_fn):
    # improve_sol_fn maps model to a solution
    # should be state-less
    super(EvalHeur, self).__init__()
    self.improve_sol_fn = improve_sol_fn
    self._step = 0
    # collects (obj_val, step, is_heuristic_improvement) tuples whenever
    # obj_val changes.
    # is_heuristic_improvement is true if the improvement is from
    # the improve_sol_fn being called.
    self._obj_vals = []

  def return_fn(self, found_sol=True):
    self._step += 1
    if found_sol:
      return dict(result=SCIP_RESULT.FOUNDSOL)
    return dict(result=SCIP_RESULT.DIDNOTFIND)

  def heurexec(self, heurtiming, nodeinfeasible):
    model = self.model
    if model.getGap() > 1e10:
      return self.return_fn(False)
    scip_sol = model.getBestSol()
    prev_obj = model.getSolObjVal(scip_sol)
    if len(self._obj_vals) == 0:
      # Initialize with the starting gap.
      self._obj_vals.append((None, prev_obj, self._step, model.getGap(), None, False))
    else:
      if self._obj_vals[-1][1] > prev_obj:
        # improvement in objective occured in branch-and-bound algorithm
        self._obj_vals.append(
            (self._obj_vals[-1][1], prev_obj, self._step, model.getGap(), None, False))

    print(f'Step: {self._step}, Obj:{prev_obj}, Gap:{model.getGap()}')
    varname2var = {v.name.lstrip('t_'): v for v in model.getVars()}
    # convert scip_sol to dict
    sol_d = {n: model.getSolVal(scip_sol, v) for n, v in varname2var.items()}

    if self._step > 0:
      sol, stats = self.improve_sol_fn(model, sol_d, prev_obj, self._step)
      if sol:
        assert len(sol) >= len(sol_d), [len(sol), len(sol_d)]
        # convert sol to sol_scip
        sol_scip = model.createSol(self)
        vars_with_exception = []
        for var_name in sol:
          try:
            model.setSolVal(sol_scip, varname2var[var_name], sol[var_name])
          except Exception as e:
            vars_with_exception.append(var_name)
        if vars_with_exception:
          print(f'WARNING: Exception encountered in {len(vars_with_exception)} of {len(sol)}')
        # record the improved objective
        improved_obj = model.getSolObjVal(sol_scip)
        print(f'Prev_obj: {prev_obj} Improved_obj: {improved_obj}')
        if improved_obj < prev_obj:
          self._obj_vals.append((prev_obj, improved_obj, self._step, model.getGap(), stats, True))
          self.model.addSol(sol_scip, free=True)
        assert fabs(improved_obj - model.getSolObjVal(model.getBestSol())) <= 1e-3
        return self.return_fn(True)
    return self.return_fn(False)

  def done(self):
    self._obj_vals.append((None, None, self._step, self.model.getGap(), None, False))


def init_scip_params(
    model,
    seed,
    presolving=True,
    conflict=True,
    heur_frequency=-1,
):
  # forked from liaison/daper/milp/features.py

  seed = seed % 2147483648  # SCIP seed range

  # set up randomization
  model.setBoolParam('randomization/permutevars', True)
  model.setIntParam('randomization/permutationseed', seed)
  model.setIntParam('randomization/randomseedshift', seed)

  # Don't limit max rounds for now.
  # model.setIntParam('separating/maxrounds', 0)

  # disable presolving
  if not presolving:
    model.setIntParam('presolving/maxrestarts', 0)
    model.setIntParam('presolving/maxrounds', 0)
    model.setIntParam('presolving/maxrestarts', 0)
    model.setPresolving(SCIP_PARAMSETTING.OFF)

  # disable separating (cuts)
  model.setIntParam('separating/maxroundsroot', 0)
  model.setSeparating(SCIP_PARAMSETTING.OFF)

  # if asked, disable conflict analysis (more cuts)
  if not conflict:
    model.setBoolParam('conflict/enable', False)

  # disable pscost for branching.
  model.setParam('branching/pscost/priority', 100000000)

  # disable all heuristics.
  for k, v in model.getParams().items():
    if k.startswith('heuristics/') and k.endswith('/freq'):
      model.setParam(k, heur_frequency)
  # enable simplerounding heuristic to run at the
  # root node.
  model.setParam('heuristics/simplerounding/freq', 1)


def run_branch_and_bound_scip(m, heuristic):
  # m must be presolved before passing
  m.includeHeur(heuristic,
                "PyEvalHeur",
                "custom heuristic implemented in python to evaluate RL agent",
                "Y",
                timingmask=SCIP_HEURTIMING.BEFORENODE)

  with U.Timer() as timer:
    m.optimize()

  # collect stats
  results = ConfigDict(mip_work=m.getNNodes(),
                       n_cuts=m.getNCuts(),
                       n_cuts_applied=m.getNCutsApplied(),
                       n_lps=m.getNLPs(),
                       pre_solving_time=m.getPresolvingTime(),
                       solving_time=m.getSolvingTime(),
                       time_elapsed=timer.to_seconds())
  # m.freeProb()
  return results
