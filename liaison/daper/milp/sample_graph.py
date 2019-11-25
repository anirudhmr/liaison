import argparse
import functools
import os
import pickle
from multiprocessing.pool import ThreadPool

from liaison.daper.milp.dataset import MILP
from liaison.daper.milp.generate_graph import generate_instance

parser = argparse.ArgumentParser()
parser.add_argument('--out_file', type=str, required=True)
parser.add_argument('--problem_type',
                    type=str,
                    required=True,
                    help='Options: cauction, facilities')
parser.add_argument('--problem_size', type=int, required=True)
parser.add_argument('--time_limit', type=int, default=None)
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()


def main():

  milp = MILP()
  milp.seed = args.seed
  milp.problem_type = args.problem_type
  mip = milp.mip = generate_instance(args.problem_type, args.problem_size,
                                     args.seed)

  import cplex
  solver = cplex.Cplex()
  if args.time_limit:
    solver.parameters.timelimit.set(args.time_limit)

  mip.add_to_cplex_solver(solver)

  print('Number of variables = ', solver.variables.get_num())
  print('Number of constraints = ', solver.linear_constraints.get_num())

  solver.solve()

  print(solver.solution.get_status())
  print('Objective value = ', solver.solution.get_objective_value())

  var_names = list(mip.varname2var.keys())
  vals = solver.solution.get_values(var_names)
  assignment = {var: val for var, val in zip(var_names, vals)}

  # record optimal
  milp.optimal_objective = solver.solution.get_objective_value()
  milp.optimal_solution = assignment
  milp.is_optimal = (solver.solution.get_status() == 0)

  # record feasible solution
  num_feasible_sols = solver.solution.pool.get_num()
  init_soln = solver.solution.pool.get_values(num_feasible_sols - 1)
  init_obj_val = solver.solution.pool.get_objective_value(num_feasible_sols -
                                                          1)
  assert sorted(var_names) == sorted(list(solver.variables.get_names()))

  milp.feasible_objective = init_obj_val
  # convert to dict format from var_name -> value.
  milp.feasible_solution = {
      var: val
      for var, val in zip(solver.variables.get_names(), init_soln)
  }

  os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
  with open(args.out_file, 'wb') as f:
    pickle.dump(milp, f)


if __name__ == '__main__':
  main()
