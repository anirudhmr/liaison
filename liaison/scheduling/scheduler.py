import itertools
from collections import namedtuple

from ortools.linear_solver import pywraplp

from .mip_primitives import Constraint, Objective, Process, Variable


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
    self._solver = pywraplp.Solver(
        'Liaison Schedule Solver',
        pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    self._wunits = []
    self._assignment_vars = []
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
    for proc_id, process in enumerate(wu):
      proc = Process(proc_id, process.cpu_cost, process.mem_cost,
                     process.gpu_compute_cost, process.gpu_mem_cost)
      ass_vars = self._add_process(proc, len(self._wunits) - 1, proc_id)

      for ass_var, constraint in zip(ass_vars, self._mem_constraints):
        constraint.add_term(ass_var.name, proc.mem_cost)

      self._wunits[-1].append(proc)
      self._assignment_vars[-1].append(ass_vars)

  def _get_objective(self):
    # first lets do the overload part of the objective
    # For server i, the term is [L_i - C_i]+
    overload_obj = Objective()
    overload_ds = []
    for server_id, server in enumerate(self._servers):
      # Create helper variable d such that
      # d = [L_i - C_i]+ using the following constraints
      # d >= 0 and d >= L_i - C_i
      # (expressed below equivalently as L_i - d <= C_i)
      # and objective is to minimize d
      d = Variable('server_%d/overload_helper_var' % server_id, 0, None,
                   self._variable_constraints)
      self._varnames2var['server_%d/overload_helper_var' % server_id] = d
      overload_obj.add_term(d.name, 1)  # Minimize d
      c = Constraint(sense='LE', rhs=server.cpu)
      c.add_term(d.name, -1)
      for wid, wunit in enumerate(self._wunits):
        for ass_var, proc in zip(self._assignment_vars[wid], wunit):
          c.add_term(ass_var[server_id].name, proc.cpu_cost)
      self._misc_constraints.append(c)
      overload_ds.append(d)

    # Next let's do load balancing.
    # For server i, the overload is defined as [L_i - C_i]+
    # We seek to minimize max_overload - min_overload as a
    # way to load balance the excess overloads.
    load_balancing_obj = Objective()
    d_max = Variable('load_balancing_max_helper', 0, None,
                     self._variable_constraints)
    d_min = Variable('load_balancing_min_helper', 0, None,
                     self._variable_constraints)
    self._varnames2var['load_balancing_max_helper'] = d_max
    self._varnames2var['load_balancing_min_helper'] = d_min
    load_balancing_obj.add_term(d_max.name, 1)
    load_balancing_obj.add_term(d_min.name, -1)

    for server_id, server in enumerate(self._servers):
      # d is [L_i - C_i]+ calculated from the previous step.
      d = overload_ds[server_id]
      # add d_min <= d
      c = Constraint(sense='LE', rhs=0)
      c.add_term(d_min.name, 1)
      c.add_term(d.name, -1)
      self._misc_constraints.append(c)

      # add d_max >= d
      c = Constraint(sense='GE', rhs=0)
      c.add_term(d_max.name, 1)
      c.add_term(d.name, -1)
      self._misc_constraints.append(c)

    # Now, Let's do work unit consolidation.
    # If processes of a work unit in total
    # use n servers add a penalty proportional to n
    work_unit_consolidation_obj = Objective()
    for wid, wunit in enumerate(self._wunits):
      for server_id, server in enumerate(self._servers):
        d = Variable('wu_%d/server_%d_consolidation_helper' % (wid, server_id),
                     0, 1, self._variable_constraints)
        self._varnames2var['wu_%d/server_%d_consolidation_helper' %
                           (wid, server_id)] = d
        work_unit_consolidation_obj.add_term(d.name, 1)
        for process_vars in self._assignment_vars[wid]:
          # d >= x_i
          c = Constraint(sense='GE', rhs=0)
          c.add_term(d.name, 1)
          c.add_term(process_vars[server_id].name, -1)
          self._misc_constraints.append(c)

    final_objective = Objective.combine_objectives(
        [overload_obj, load_balancing_obj, work_unit_consolidation_obj], [
            self._overload_obj_coeff, self._load_balancing_obj_coeff,
            self._wu_consolidation_obj_coeff
        ])
    self._min_objective += len(self._wunits) * self._wu_consolidation_obj_coeff
    return final_objective

  def solve(self):
    obj = self._get_objective()
    solver = self._solver
    varnames2ortoolsvar = {
        k: v.convert_to_ortools_solver_format(solver)
        for k, v in self._varnames2var.items()
    }
    obj = obj.add_to_ortools_solver(solver, varnames2ortoolsvar)
    for constraint in itertools.chain(self._mem_constraints +
                                      self._misc_constraints +
                                      self._assignment_constraints +
                                      self._variable_constraints):
      constraint.add_to_ortools_solver(solver, varnames2ortoolsvar)

    print('Number of variables =', solver.NumVariables())
    print('Number of constraints =', solver.NumConstraints())

    result_status = solver.Solve()
    assert result_status == pywraplp.Solver.OPTIMAL
    print('Objective value =', obj.Value() - self._min_objective)

    assignment = []
    for wu_vars in self._assignment_vars:
      assignment.append([])
      for process_vars in wu_vars:
        assignment[-1].append([])
        for server_var in process_vars:
          server_var = varnames2ortoolsvar[server_var.name]
          assignment[-1][-1].append(int(server_var.solution_value()))
    return assignment
