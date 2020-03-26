import pickle

from absl import app
from liaison.daper.milp.scip_mip import SCIPMIPInstance
from liaison.env.utils.rins import get_sample
from pyscipopt import Model


def main(_):
  milp = get_sample('milp-cauction-100-filtered', 'train', 0)
  val = milp.feasible_solution['x1']
  mip = SCIPMIPInstance.fromMIPInstance(milp.mip)
  mip2 = mip.fix({'t_x1': val})
  assert 't_x1' not in mip2.varname2var
  mip2.get_feasible_solution()


if __name__ == '__main__':
  app.run(main)
