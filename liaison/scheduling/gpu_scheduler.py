import itertools

from ortools.linear_solver import pywraplp

from .mip_primitives import (Constraint, Expression, Objective, Variable,
                             MIPTracker, compute_min, compute_max,
                             compute_relu)
from .gpu_scheduler_classes import WorkUnit, Server


class LiaisonGPUScheduler:

  def __init__(self,
               servers,
               wunits,
               scheduling_constraints=None,
               colocation_constraints=None,
               overload_obj_coeff=1,
               load_balancing_obj_coeff=1,
               wu_consolidation_obj_coeff=0.25):
    """
    Args:
      servers -> [dict(gpu_compute=List, gpu_mem=List, mem=float)]
      wunits -> [[dict(gpu_compute_cost=List, gpu_mem_cost=List, mem=float)]]
      scheduling_constraints -> Dict[Tuple[wid, pid] -> List[server_id]]
      colocation_constraints -> List[List[Tuple[wid, pid]]]
    """
    self._overload_obj_coeff = overload_obj_coeff
    self._load_balancing_obj_coeff = load_balancing_obj_coeff
    self._wu_consolidation_obj_coeff = wu_consolidation_obj_coeff
    self.mip = MIPTracker()
    self._servers = [
        Server(i, server.gpu_compute, server.gpu_mem, server.mem, self.mip,
               colocation_constraints) for i, server in enumerate(servers)
    ]
    self._wunits = [
        WorkUnit(i, proc_specs) for i, proc_specs in enumerate(wunits)
    ]
    for i, wunit in enumerate(self._wunits):
      # filter out the constraints for this work unit.
      # and convert from Tuple[Tuple[wid, pid], server_id] to Tuple[pid, server_id]
      wu_sched_const = {
          pid: sids
          for (wid, pid), sids in scheduling_constraints.items() if wid == i
      }
      wunit.send_requests(self._servers, self.mip, wu_sched_const)

  def _get_objective(self):

    # utilization obj
    util_objs = []
    for server in self._servers:
      util_objs.append(server.get_utilization_obj())

    util_obj = Objective.combine_objectives(util_objs, [1] * len(util_objs))
    del util_objs

    # load balancing obj
    lb_objs = []
    for server in self._servers:
      lb_objs.append(server.get_load_balancing_obj())

    lb_obj = Objective.combine_objectives(lb_objs, [1] * len(lb_objs))
    del lb_objs

    # wu consolidation obj
    objs = []
    for wu in self._wunits:
      objs.append(wu.get_wu_consolidation_obj(self.mip))
    wu_consol_obj = Objective.combine_objectives(objs, [1] * len(objs))
    del objs

    final_objective = Objective.combine_objectives(
        [util_obj, lb_obj, wu_consol_obj], [
            self._overload_obj_coeff, self._load_balancing_obj_coeff,
            self._wu_consolidation_obj_coeff
        ])
    return final_objective

  def _get_all_constraints(self):
    return self.mip.constraints

  def solve_cplex(self, time_limit=None):
    import cplex
    obj = self._get_objective()
    solver = cplex.Cplex()
    self.solver = solver
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
    print('Objective value =', solver.solution.get_objective_value())

    wu_assignments = []
    for wunit in self._wunits:
      assignment = wunit.get_assignment_vars(solver)
      wu_assignments.append(assignment)

    return wu_assignments
