import threading

import cplex
from absl.testing import absltest
from liaison.daper.milp.generate_graph import generate_instance
from pyscipopt import Model, multidict, quicksum

lock = threading.Lock()


class ScipTest(absltest.TestCase):

  def _setup(self):
    problem = generate_instance('cauction', 20, 42)
    return problem

  def testSCIP(self):
    prob = self._setup()
    model = Model("test")
    prob.add_to_scip_solver(model)
    model.optimize()
    with lock:
      print('SCIP Objective value: ', model.getObjVal())

  def testCPLEX(self):
    prob = self._setup()
    model = cplex.Cplex()
    prob.add_to_cplex_solver(model)
    model.solve()

    with lock:
      print('CPLEX Objective value: ', model.solution.get_objective_value())


if __name__ == '__main__':
  absltest.main()
