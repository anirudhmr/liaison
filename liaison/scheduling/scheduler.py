import itertools
import copy
from collections import namedtuple

from ortools.linear_solver import pywraplp

from .mip_primitives import Constraint, Objective, Process, Variable, Expression


class LiaisonScheduler:

  def __init__(self,
               servers,
               overload_obj_coeff=1,
               load_balancing_obj_coeff=1,
               wu_consolidation_obj_coeff=10):
    """
    Args:
      servers -> [dict(cpu=float, mem=float, gpu_compute=List, gpu_mem=List)]
    """
    self._overload_obj_coeff = overload_obj_coeff
    self._load_balancing_obj_coeff = load_balancing_obj_coeff
    self._wu_consolidation_obj_coeff = wu_consolidation_obj_coeff
    self._servers = servers
    self._wunits = []
    self._assignment_vars = []
    self._gpu_assignment_vars = []
    self._varnames2var = dict()

    # Minimum achievable objective value if
    # conditions are terribly optimistic
    self._min_objective = 0

    # declare constraints
    self._mem_constraints = []
    for server in self._servers:
      self._mem_constraints.append(Constraint('LE', server.mem))

    # Constraints for assignment variables
    self._assignment_constraints = []

    # Constraints when creating variables
    self._variable_constraints = []

    # Constraints used to help form certain kinds of objectives
    self._misc_constraints = []

  def _add_process(self, proc, wu_id, proc_id):
    """Creates assignment variables for the process."""
    assignment_vars = []
    c = Constraint(sense='E', rhs=1)
    for i, server in enumerate(self._servers):
      var_name = 'wu-%d/proc-%d/server-%d/assignment_var' % (wu_id, proc_id, i)
      var = Variable(var_name, 0, 1, self._variable_constraints)
      self._varnames2var[var_name] = var
      assignment_vars.append(var)
      c.add_term(var_name, 1)

    self._assignment_constraints.append(c)
    return assignment_vars

  def add_work_unit(self, wu):
    self._wunits.append([])
    self._assignment_vars.append([])
    self._gpu_assignment_vars.append([])
    for proc_id, process in enumerate(wu):
      proc = Process(proc_id, process.cpu_cost, process.mem_cost,
                     process.gpu_compute_cost, process.gpu_mem_cost)
      ass_vars = self._add_process(proc, len(self._wunits) - 1, proc_id)

      for ass_var, constraint in zip(ass_vars, self._mem_constraints):
        constraint.add_term(ass_var.name, proc.mem_cost)

      self._wunits[-1].append(proc)
      self._assignment_vars[-1].append(ass_vars)

  def _compute_max(self, exprs, var_name, Ls, Us):
    """
      Let U' = max(Us)
      expr[i] must lie between Ls[i] and Us[i]
      y >= expr_i
      y <= expr_i + (U' - L_i)* (1 - d_i)
      sum d_i = 1
    """
    U1 = max(Us)
    ds = []
    for i in range(len(exprs)):
      d = Variable('%s/helper/d%d' % (var_name, i), 0, 1,
                   self._variable_constraints)
      self._varnames2var['%s/helper/d%d' % (var_name, i)] = d
      ds.append(d)

    y = Variable(var_name)
    self._varnames2var[var_name] = y

    # y >= expr_i
    # expr_i - y <= 0
    for expr in exprs:
      c = expr.to_constraint('LE', 0)
      c.add_term(y.name, -1)
      self._misc_constraints.append(c)

    # y <= expr + (U' - L_i)* (1 - d2)
    # expr - y + (U' - L_i)* (1 - d2) >= 0
    # expr - y + (L_i - U')* d2 >= L_i - U'
    for expr, d, l in zip(exprs, ds, Ls):
      c = expr.to_constraint('GE', l - U1)
      c.add_terms([y.name, d.name], [-1, (l - U1)])
      self._misc_constraints.append(c)

    # sum d_i = 1
    c = Constraint('E', 1)
    c.add_terms([d.name for d in ds], [1] * len(ds))
    self._misc_constraints.append(c)
    return y

  def _compute_min(self, exprs, var_name, Ls, Us):
    """
      Let L' = min(Ls)
      expr[i] must lie between Ls[i] and Us[i]
      y <= expr_i
      y >= expr_i - (U_i - L')* (1 - d_i)
      sum d_i = 1
    """
    L1 = min(Ls)
    ds = []
    for i in range(len(exprs)):
      d = Variable('%s/helper/d%d' % (var_name, i), 0, 1,
                   self._variable_constraints)
      self._varnames2var['%s/helper/d%d' % (var_name, i)] = d
      ds.append(d)

    y = Variable(var_name)
    self._varnames2var[var_name] = y

    # y <= expr_i
    # expr_i - y >= 0
    for expr in exprs:
      c = expr.to_constraint('GE', 0)
      c.add_term(y.name, -1)
      self._misc_constraints.append(c)

    # y >= expr - (U_i - L')* (1 - d_i)
    # expr - y - (U_i - L')* (1 - d_i) <= 0
    # expr - y + (U_i - L')* d_i <= U_i - L'
    for expr, d, u in zip(exprs, ds, Us):
      c = expr.to_constraint('LE', u - L1)
      c.add_terms([y.name, d.name], [-1, (u - L1)])
      self._misc_constraints.append(c)

    # sum d_i = 1
    c = Constraint('E', 1)
    c.add_terms([d.name for d in ds], [1] * len(ds))
    self._misc_constraints.append(c)
    return y

  def _compute_relu(self, expr, var_name, L, U):
    return self._compute_max([expr, Expression(0)], var_name, [L, 0], [U, 0])

  def _get_overload_components(self):
    """Returns [L_i - C_i]+ for each server i"""
    overload_ds = []
    max_load = sum([proc.cpu_cost for wunit in self._wunits for proc in wunit])
    for server_id, server in enumerate(self._servers):
      # Compute the expression L_i - C_i
      expr = Expression()
      expr.add_constant(-server.cpu)
      for wid, wunit in enumerate(self._wunits):
        for ass_var, proc in zip(self._assignment_vars[wid], wunit):
          expr.add_term(ass_var[server_id].name, proc.cpu_cost)

      d = self._compute_relu(expr, 'server_%d/overload_helper_var' % server_id,
                             -server.cpu, max_load - server.cpu)
      overload_ds.append(d)
    return overload_ds

  def _get_objective(self):
    # first lets do the overload part of the objective
    # For server i, the term is [L_i - C_i]+
    max_load = sum([proc.cpu_cost for wunit in self._wunits for proc in wunit])
    overload_ds = self._get_overload_components()

    overload_obj = Objective()
    for d in overload_ds:
      overload_obj.add_term(d.name, 1)

    # Next let's do load balancing.
    # For server i, the overload is defined as [L_i - C_i]+
    # We seek to minimize max_overload - min_overload as a
    # way to load balance the excess overloads.
    d_max = self._compute_max(
        [d.to_expression() for d in overload_ds], 'load_balancing_max_helper',
        [0] * len(overload_ds),
        [max_load - server.cpu for server in self._servers])

    d_min = self._compute_min(
        [d.to_expression() for d in overload_ds], 'load_balancing_min_helper',
        [0] * len(overload_ds),
        [max_load - server.cpu for server in self._servers])

    load_balancing_obj = Objective()
    load_balancing_obj.add_term(d_max.name, 1)
    load_balancing_obj.add_term(d_min.name, -1)

    # Now, Let's do work unit consolidation.
    # If processes of a work unit in total
    # use n servers add a penalty proportional to n
    work_unit_consolidation_obj = Objective()
    for wid, wunit in enumerate(self._wunits):
      # get count of the number of distinct servers
      # used by the work unit.
      for server_id, server in enumerate(self._servers):
        e = Expression()
        for process_vars in self._assignment_vars[wid]:
          e.add_term(process_vars[server_id].name, 1)

        v = self._compute_min([Expression(1), e],
                              'wu_%d/server_%d_consolidation_helper' %
                              (wid, server_id), [1, 0], [1, len(wunit)])
        work_unit_consolidation_obj.add_term(v.name, 1)

    final_objective = Objective.combine_objectives(
        [overload_obj, load_balancing_obj, work_unit_consolidation_obj], [
            self._overload_obj_coeff, self._load_balancing_obj_coeff,
            self._wu_consolidation_obj_coeff
        ])
    self._min_objective += len(self._wunits) * self._wu_consolidation_obj_coeff
    return final_objective

  def _get_all_constraints(self, ignore_variable_constraints=False):
    constraints = self._mem_constraints + self._misc_constraints + self._assignment_constraints
    if ignore_variable_constraints:
      return constraints
    else:
      return constraints + self._variable_constraints

  def solve_or_tools(self, time_limit=None):
    """TODO: Implement time_limit."""
    obj = self._get_objective()
    solver = pywraplp.Solver('Liaison Schedule Solver',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    varnames2ortoolsvar = {
        k: v.convert_to_ortools_solver_format(solver)
        for k, v in self._varnames2var.items()
    }
    for constraint in self._get_all_constraints(
        ignore_variable_constraints=True):
      constraint.add_to_ortools_solver(solver, varnames2ortoolsvar)
    obj = obj.add_to_ortools_solver(solver, varnames2ortoolsvar)

    print('Number of variables =', solver.NumVariables())
    print('Number of constraints =', solver.NumConstraints())

    result_status = solver.Solve()
    assert result_status == pywraplp.Solver.OPTIMAL
    print('Objective value =', obj.Value())

    assignment = []
    for wu_vars in self._assignment_vars:
      assignment.append([])
      for process_vars in wu_vars:
        assignment[-1].append([])
        for server_var in process_vars:
          server_var = varnames2ortoolsvar[server_var.name]
          assignment[-1][-1].append(int(server_var.solution_value()))
    return assignment

  def solve_cplex(self, time_limit=None):
    import cplex
    obj = self._get_objective()
    solver = cplex.Cplex()
    if time_limit:
      solver.parameters.timelimit.set(time_limit)
    for v in self._varnames2var.values():
      v.add_to_cplex_solver(solver)
    for constraint in self._get_all_constraints(
        ignore_variable_constraints=True):
      constraint.add_to_cplex_solver(solver)
    obj.add_to_cplex_solver(solver)

    print('Number of variables =', solver.variables.get_num())
    print('Number of constraints =', solver.linear_constraints.get_num())

    solver.solve()
    print(solver.solution.get_status())
    print('Objective value =',
          solver.solution.get_objective_value() - self._min_objective)

    assignment = []
    for wu_vars in self._assignment_vars:
      assignment.append([])
      for process_vars in wu_vars:
        assignment[-1].append(
            solver.solution.get_values(
                [server_var.name for server_var in process_vars]))
    return assignment
