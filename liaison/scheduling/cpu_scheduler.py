"""
  Schedules multiple work units without
  violating memory constraints
"""
import itertools
from collections import namedtuple

from ortools.linear_solver import pywraplp

from .mip_primitives import (Constraint, Expression, Objective, Variable,
                             MIPTracker, compute_min, compute_max,
                             compute_relu)


class Process:

  def __init__(self, id, cpu_cost, mem_cost):
    self._id = id
    self.cpu_cost = cpu_cost
    self.mem_cost = mem_cost


class LiaisonCPUScheduler:

  def __init__(self,
               servers,
               scheduling_constraints=[],
               colocation_constraints={},
               overload_obj_coeff=1,
               load_balancing_obj_coeff=1,
               wu_consolidation_obj_coeff=10):
    """
    Args:
      servers -> [dict(cpu=float, mem=float)]
      scheduling_constraints -> Dict[Tuple[wid, pid] -> List[server_id]
      colocation_constraints -> List[List[Tuple[wid, pid]]]
    """
    self._overload_obj_coeff = overload_obj_coeff
    self._load_balancing_obj_coeff = load_balancing_obj_coeff
    self._wu_consolidation_obj_coeff = wu_consolidation_obj_coeff
    self._scheduling_constraints = scheduling_constraints
    self._colocation_constraints = colocation_constraints
    self._servers = servers
    self._wunits = []
    self._assignment_vars = []
    self.mip = MIPTracker()

    # Minimum achievable objective value if
    # conditions are terribly optimistic
    self._min_objective = 0

    # declare constraints
    self._mem_constraints = []
    for server in self._servers:
      self._mem_constraints.append(Constraint('LE', server.mem))

    # Constraints for assignment variables
    self._assignment_constraints = []

    # (wid, pid) -> [assignment_vars] for the corresponding process.
    self._vars_for_colocation_constraints = {}

  def _add_process(self, proc, wu_id, proc_id):
    """Creates assignment variables for the process."""
    assignment_vars = []
    c = Constraint(sense='E', rhs=1)
    for i, server in enumerate(self._servers):
      var_name = 'wu-%d/proc-%d/server-%d/assignment_var' % (wu_id, proc_id, i)
      var = self.mip.new_variable(var_name, 0, 1)
      assignment_vars.append(var)
      c.add_term(var_name, 1)

    self._vars_for_colocation_constraints[(wu_id, proc_id)] = assignment_vars
    self._assignment_constraints.append(c)

    if (wu_id, proc_id) in self._scheduling_constraints:
      c = Constraint(sense='E', rhs=1)
      for sid in self._scheduling_constraints[(wu_id, proc_id)]:
        c.add_term(assignment_vars[sid].name, 1)
      self._assignment_constraints.append(c)

    for coloc_constraint in self._colocation_constraints:
      if (wu_id, proc_id) in coloc_constraint:
        for wid, pid in coloc_constraint:
          if wid != wu_id and pid != proc_id:
            for v1, v2 in zip(
                self._vars_for_colocation_constraints[(wid, pid)],
                self._vars_for_colocation_constraints[(wu_id, proc_id)]):
              c = Constraint(sense='E', rhs=0)
              c.add_term(v1.name, 1)
              c.add_term(v2.name, -1)
              self._assignment_constraints.append(c)

    return assignment_vars

  def add_work_unit(self, wu):
    self._wunits.append([])
    self._assignment_vars.append([])
    for proc_id, process in enumerate(wu):
      proc = Process(proc_id, int(process.cpu_cost), int(process.mem_cost))

      ass_vars = self._add_process(proc, len(self._wunits) - 1, proc_id)

      for ass_var, constraint in zip(ass_vars, self._mem_constraints):
        constraint.add_term(ass_var.name, proc.mem_cost)

      self._wunits[-1].append(proc)
      self._assignment_vars[-1].append(ass_vars)

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

      d = compute_relu(expr, 'server_%d/overload_helper_var' % server_id,
                       -server.cpu, max_load - server.cpu, self.mip)
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
    d_max = compute_max(
        [d.to_expression() for d in overload_ds], 'load_balancing_max_helper',
        [0] * len(overload_ds),
        [max(0, max_load - server.cpu) for server in self._servers], self.mip)

    d_min = compute_min(
        [d.to_expression() for d in overload_ds], 'load_balancing_min_helper',
        [0] * len(overload_ds),
        [max(0, max_load - server.cpu) for server in self._servers], self.mip)

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

        v = compute_min([Expression(1), e],
                        'wu_%d/server_%d_consolidation_helper' %
                        (wid, server_id), [1, 0], [1, len(wunit)], self.mip)
        work_unit_consolidation_obj.add_term(v.name, 1)

    final_objective = Objective.combine_objectives(
        [overload_obj, load_balancing_obj, work_unit_consolidation_obj], [
            self._overload_obj_coeff, self._load_balancing_obj_coeff,
            self._wu_consolidation_obj_coeff
        ])
    self._min_objective += len(self._wunits) * self._wu_consolidation_obj_coeff
    return final_objective

  def _get_all_constraints(self):
    constraints = self._mem_constraints + self._assignment_constraints + self.mip.constraints
    return constraints

  def solve_or_tools(self, time_limit=None):
    """TODO: Implement time_limit."""
    obj = self._get_objective()
    solver = pywraplp.Solver('Liaison Schedule Solver',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    varnames2ortoolsvar = {
        k: v.convert_to_ortools_solver_format(solver)
        for k, v in self.mip.varnames2var.items()
    }
    for constraint in self._get_all_constraints():
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
    for v in self.mip.varnames2var.values():
      v.add_to_cplex_solver(solver)
    for constraint in self._get_all_constraints():
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
        l = solver.solution.get_values(
            [server_var.name for server_var in process_vars])
        assert l.count(1) == 1
        assignment[-1].append(l.index(1))
    return assignment
