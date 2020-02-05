import copy

import pyscipopt as scip
from liaison.daper.milp.dataset import MIPInstance
from liaison.daper.milp.features import (get_features_from_scip_model,
                                         init_scip_params)

# Interface for MIP instances built on top of SCIP


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
    self.varname2var = {v.name for v in self.vars}
    self.model = model

  @staticmethod
  def fromMIPInstance(m):
    solver = get_model()
    m.add_to_scip_solver(solver)
    return SCIPMIPInstance(solver)

  def get_features(self):
    # create a copy to avoid polluting the current one.
    # returns constraint_features, edge_features, variable_features
    m = scip.Model(sourceModel=self.model)
    return get_features_from_scip_model(m)

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
      assert v in self.varname2var
      var = self.varname2var[v]
      assert var.getLbGlobal() <= val and val <= var.getUbGlobal()

    # copy the original model
    fixed_model = scip.Model(sourceModel=self.model)
    # fix the upper and lower bounds for the variables.
    for v, val in fixed_ass.items():
      fixed_model.chgVarLbGlobal(val)
      fixed_model.chgVarUbGlobal(val)
    m = SCIPMIPInstance(fixed_model)
    return m
