import copy

import pyscipopt as scip
from liaison.daper.milp.dataset import MIPInstance
from liaison.daper.milp.features import (get_features_from_scip_model,
                                         init_scip_params)

# Interface for MIP instances built on top of SCIP
EPSILON = 1e-4


def get_model():
  m = scip.Model()
  m.hideOutput()
  m.setIntParam('display/verblevel', 0)
  init_scip_params(m, seed=42)
  m.setIntParam('timing/clocktype', 2)
  m.setRealParam('limits/time', 120)
  m.setParam('limits/nodes', 1)
  return m


class SCIPMIPInstance:

  def __init__(self, model):
    model.presolve()
    self.vars = list(model.getVars(transformed=True))
    self.varname2var = {v.name: v for v in self.vars}
    self.model = model

  @staticmethod
  def fromMIPInstance(m):
    solver = get_model()
    m.add_to_scip_solver(solver)
    return SCIPMIPInstance(solver)

  def get_features(self):
    # create a copy to avoid polluting the current one.
    # returns constraint_features, edge_features, variable_features
    m = self._copy_model()
    return get_features_from_scip_model(m)

  def _copy_model(self):
    return scip.Model(sourceModel=self.model, origcopy=True)

  def fix(self, fixed_ass, relax_integral_constraints=False):
    """
      Args:
        fixed_vars_to_values: variables to fix and their values to fix to.
        integral_relax: If true return an lp with all integral constraints on
                        all variables relaxed.
      Returns:
        Leaves the current mipinstance unchanged (immutable call).
        Returns a new mipinstance with fixes and relaxations made.
    """
    for v, val in fixed_ass.items():
      v = 't_' + v
      assert v in self.varname2var
      var = self.varname2var[v]
      assert var.getLbGlobal() - EPSILON <= val and val <= var.getUbGlobal() + EPSILON

    # copy the original model
    fixed_model = self._copy_model()
    fixed_model_vars = list(fixed_model.getVars(transformed=True))
    fixed_model_varname2var = {v.name.lstrip('t_'): v for v in fixed_model_vars}

    # fix the upper and lower bounds for the variables.
    for v, val in fixed_ass.items():
      var = fixed_model_varname2var[v]
      fixed_model.chgVarLbGlobal(var, val - EPSILON)
      fixed_model.chgVarUbGlobal(var, val + EPSILON)

    if relax_integral_constraints:
      for v in fixed_model.getVars():
        fixed_model.chgVarType(v, 'CONTINUOUS')

    m = SCIPMIPInstance(fixed_model)
    return m

  def get_feasible_solution(self):
    # get any feasible solution.

    fixed_model = self._copy_model()
    # No objective -- only feasibility
    fixed_model.setObjective(scip.Expr())

    fixed_model.optimize()

    ass = {'t_' + var.name: fixed_model.getVal(var) for var in fixed_model.getVars()}
    return ass

  def get_scip_model(self):
    return self._copy_model()
