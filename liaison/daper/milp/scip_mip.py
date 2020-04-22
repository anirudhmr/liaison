import copy

import pyscipopt as scip
from liaison.daper.milp.dataset import MIPInstance
from liaison.daper.milp.features import (get_features_from_scip_model,
                                         init_scip_params)
from liaison.daper.milp.scip_utils import del_scip_model

# Interface for MIP instances built on top of SCIP
EPSILON = 1e-4


def get_model():
  m = scip.Model()
  m.hideOutput()
  m.setIntParam('display/verblevel', 0)
  init_scip_params(m, seed=42)
  return m


class SCIPMIPInstance:

  def __init__(self, model):
    model.presolve()
    self.vars = list(model.getVars(transformed=True))
    self.varname2var = {v.name: v for v in self.vars}
    self.originalVarBounds = {v.name: (v.getLbGlobal(), v.getUbGlobal()) for v in self.vars}
    self.originalVarTypes = {v.name: v.vtype() for v in self.vars}
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
    ret = get_features_from_scip_model(m)
    del_scip_model(m)
    return ret

  def _copy_model(self):
    return scip.Model(sourceModel=self.model, origcopy=True)

  def fix(self, fixed_ass, relax_integral_constraints=False, scip_model=None):
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

    if scip_model is None:
      # copy the original model
      fixed_model = self._copy_model()
    else:
      fixed_model = scip_model

    fixed_model_vars = list(fixed_model.getVars(transformed=True))
    fixed_model_varname2var = {v.name.lstrip('t_'): v for v in fixed_model_vars}

    for v, var in self.varname2var.items():
      fixed_model_var = fixed_model_varname2var[v.lstrip('t_')]

      if v.lstrip('t_') in fixed_ass:
        l = u = fixed_ass[v.lstrip('t_')]
      else:
        # set all other variable bounds to the original
        l, u = self.originalVarBounds[v]
        fixed_model.chgVarType(fixed_model_var, self.originalVarTypes[v])

      fixed_model.chgVarLbGlobal(fixed_model_var, l - EPSILON)
      fixed_model.chgVarUbGlobal(fixed_model_var, u + EPSILON)

    if relax_integral_constraints:
      for v in fixed_model.getVars():
        fixed_model.chgVarType(v, 'CONTINUOUS')

    return fixed_model

  def get_feasible_solution(self):
    # get any feasible solution.

    fixed_model = self._copy_model()
    # No objective -- only feasibility
    fixed_model.setObjective(scip.Expr())

    fixed_model.optimize()
    assert fixed_model.getStatus() == 'optimal', fixed_model.getStatus()

    ass = {'t_' + var.name: fixed_model.getVal(var) for var in fixed_model.getVars()}
    del_scip_model(fixed_model)
    return ass

  def get_scip_model(self):
    return self._copy_model()

  def __del__(self):
    del_scip_model(self.model)
