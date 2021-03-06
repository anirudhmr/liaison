import threading

import cplex
from absl.testing import absltest
from liaison.daper.milp.generate_graph import generate_instance
from liaison.daper.milp.primitives import IntegerVariable, MIPInstance
from pyscipopt import Model, multidict, quicksum

lock = threading.Lock()


class ScipTest(absltest.TestCase):

  def _setup(self):
    m = MIPInstance()
    obj = m.get_objective()
    # maximize 2x -3y + 5z
    obj.add_term('x', -2)
    obj.add_term('y', 3)
    obj.add_term('z', -5)

    # constraints
    # x + y >= 3
    c = m.new_constraint('GE', 3, name='x + y >= 3')
    c.add_terms(['x', 'y'], [1, 1])
    # x - z <= 4
    c = m.new_constraint('LE', 4, name='x - z <= 4')
    c.add_terms(['x', 'z'], [1, -1])
    # y + z <= 6
    c = m.new_constraint('LE', 6, name='y + z <= 6')
    c.add_terms(['y', 'z'], [1, 1])

    m.add_variable(IntegerVariable('x'))
    # y <= 4
    m.add_variable(IntegerVariable('y', upper_bound=4))
    # z <= 0
    m.add_variable(IntegerVariable('z', upper_bound=0))
    return m

  def _get_scip(self):
    model = Model()
    return model

  def testRelax2(self):
    full_sol = dict(x=4, y=-1, z=0)
    for case in [
        None,
        dict(),
        dict(x=4),
        dict(y=-1),
        dict(z=0),
        dict(x=4, y=-1),
        dict(x=4, z=0),
        dict(y=-1, z=0)
    ]:
      m = self._setup()
      if case: m = m.unfix(case, integral_relax=True)
      solver = self._get_scip()
      m.add_to_scip_solver(solver)
      solver.optimize()
      assert solver.getObjVal() == -11.
      sol = {var.name: solver.getVal(var) for var in solver.getVars()}
      if case:
        assert len(sol) == 3 - len(case)
      for k, v in sol.items():
        self.assertEqual(v, full_sol[k])


if __name__ == '__main__':
  absltest.main()
