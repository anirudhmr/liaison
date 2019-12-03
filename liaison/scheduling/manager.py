from collections import namedtuple

import liaison.utils as U
from liaison.scheduling import LiaisonCPUScheduler, LiaisonGPUScheduler

Server = namedtuple('Server', ['name', 'cpu', 'mem', 'gpu_compute', 'gpu_mem'])
Process = namedtuple(
    'Process',
    ['name', 'cpu_cost', 'mem_cost', 'gpu_compute_cost', 'gpu_mem_cost'])
GPU_MODEL_TO_SCALE = {
    'K80': 1,
    'V100': 2,
    'P100': 2,
    'Titan X': 1,
    'Tesla K40m': 1,
}


class ScheduleManager:

  def __init__(self,
               nodes,
               wunits,
               scheduling_constraints=None,
               colocation_constraints=None,
               time_limit=None,
               gpu_overload_obj_coeff=1,
               gpu_load_balancing_obj_coeff=1,
               gpu_wu_consolidation_obj_coeff=0.25,
               cpu_overload_obj_coeff=1,
               cpu_load_balancing_obj_coeff=1,
               cpu_wu_consolidation_obj_coeff=10):
    """
    Args:
      nodes: List of nodes
      wunits: List of wunits time_limit: Total time limit
      scheduling_constraints -> Dict[Tuple[wid, pid] -> List[server_id]]
      colocation_constraints -> List[List[Tuple[wid, pid]]]
    """
    self.time_limit = time_limit
    self.scheduling_constraints = {} if scheduling_constraints is None else scheduling_constraints
    self.colocation_constraints = [] if colocation_constraints is None else colocation_constraints

    # create servers
    self.servers = []
    for node in nodes:
      server = Server(
          node.name, int(U.relu(node.avail_cpu())),
          int(U.relu(node.avail_mem())),
          list(
              map(int, map(U.relu,
                           node.avail_gpu_compute(GPU_MODEL_TO_SCALE)))),
          list(map(int, map(U.relu, node.avail_gpu_mem()))))
      self.servers.append(server)

    # create processes in work units
    self.wunits = []
    for procs in wunits:
      self.wunits.append([])
      for proc in procs:
        p = Process(name=proc.name,
                    cpu_cost=int(proc.cpu_cost),
                    mem_cost=int(proc.mem_cost),
                    gpu_compute_cost=list(map(int, proc.gpu_compute_cost)),
                    gpu_mem_cost=list(map(int, proc.gpu_mem_cost)))
        assert p.cpu_cost is not None, p.name
        assert p.mem_cost is not None, p.name
        self.wunits[-1].append(p)

    if self._check_gpu_assignment_required():
      # first schedule gpu and get an assignment
      # assignment: List[Dict[pid] -> server_id]
      # gpu_assignment: List[Dict[pid] -> List[gpu_id]]
      gpu_server_ass, gpu_ass = self._schedule_gpu(
          gpu_overload_obj_coeff, gpu_load_balancing_obj_coeff,
          gpu_wu_consolidation_obj_coeff)

      # add previous assignments as new scheduling constraints.
      # ass: List[Dict[pid] -> server_id]
      for wid, wu_ass in enumerate(gpu_server_ass):
        for pid, sid in wu_ass.items():
          if (wid, pid) in self.scheduling_constraints:
            assert sid in self.scheduling_constraints[(wid, pid)]
          else:
            self.scheduling_constraints[(wid, pid)] = [sid]
      self._gpu_assignments = gpu_ass
    else:
      gpu_server_ass = [{} for _ in range(len(self.wunits))]
      self._gpu_assignments = [{} for _ in range(len(self.wunits))]

    self._server_assignments = self._schedule_cpu(
        self.scheduling_constraints, cpu_overload_obj_coeff,
        cpu_load_balancing_obj_coeff, cpu_wu_consolidation_obj_coeff)

    # assert that the cpu didn't violate
    # the server assignments dictated by
    # gpu assignment.
    for d1, d2 in zip(gpu_server_ass, self._server_assignments):
      for k, v in d1.items():
        assert d2[k] == v

  def _check_gpu_assignment_required(self):
    requirement = False
    for procs in self.wunits:
      for proc in procs:
        if proc.gpu_mem_cost:
          requirement = True

    availability = [any(server.gpu_mem) for server in self.servers]
    if requirement and not availability:
      raise Exception("GPU required by a process but is not available.")
    return requirement

  def _schedule_gpu(self, gpu_overload_obj_coeff, gpu_load_balancing_obj_coeff,
                    gpu_wu_consolidation_obj_coeff):
    # Now comes the part where we remove the servers
    # without GPU resources and processes without GPU
    # requirements.
    # NOte that key to understanding the following code:
    # i, l1 = zip(*filter(lambda k: k[1], enumerate(l)))
    # where l1 is filtered list of l according to the lambda function
    # i is the set of indices where elements of l1 are collected from l.

    # selected_proc_info = List[Tuple[Selected_proc_ids, Selected_procs]]
    # Selected_proc_ids = List[int]
    # Selected_procs = List[Process]
    selected_work_ids, selected_proc_info = list(
        zip(*filter(
            lambda k: k[1],
            enumerate([
                list(zip(*filter(lambda k: k[1].gpu_mem_cost, enumerate(wu))))
                for wu in self.wunits
            ]))))
    selected_server_ids, selected_servers = list(
        zip(*filter(lambda k: k[1].gpu_mem, enumerate(self.servers))))

    gpu_scheduling_constraints = {}
    for (wid, pid), sids in self.scheduling_constraints.items():
      for sid in sids:
        if sid in selected_server_ids:
          sid1 = selected_server_ids.index(sid)
          if wid in selected_work_ids:
            wid1 = selected_work_ids.index(wid)
            selected_proc_ids, selected_procs = selected_proc_info[wid1]
            if pid in selected_proc_ids:
              pid1 = selected_proc_ids.index(pid)
              if (wid1, pid1) in gpu_scheduling_constraints:
                gpu_scheduling_constraints[(wid1, pid1)].append(sid1)
              else:
                gpu_scheduling_constraints[(wid1, pid1)] = [sid1]

    gpu_colocation_constraints = []
    for colocation_l in self.colocation_constraints:
      l = []
      for wid, pid in colocation_l:
        if wid in selected_work_ids:
          wid1 = selected_work_ids.index(wid)
          selected_proc_ids, selected_procs = selected_proc_info[wid1]
          if pid in selected_proc_ids:
            pid1 = selected_proc_ids.index(pid)
            l.append((wid1, pid1))
      if l:
        gpu_colocation_constraints.append(l)

    self._gpu_scheduler = LiaisonGPUScheduler(
        selected_servers,
        list(map(lambda k: k[1], selected_proc_info)),
        overload_obj_coeff=gpu_overload_obj_coeff,
        load_balancing_obj_coeff=gpu_load_balancing_obj_coeff,
        wu_consolidation_obj_coeff=gpu_wu_consolidation_obj_coeff,
        scheduling_constraints=gpu_scheduling_constraints,
        colocation_constraints=gpu_colocation_constraints)

    solution = self._gpu_scheduler.solve_cplex(
        None if self.time_limit is None else self.time_limit / 2)

    # assignment: List[Dict[pid] -> server_id]
    assignment = [{} for _ in range(len(self.wunits))]
    # gpu_assignment: List[Dict[pid] -> List[gpu_id]]
    gpu_assignment = [{} for _ in range(len(self.wunits))]

    print('--------------------------')
    print('GPU assignment')
    print('--------------------------')
    for wu_id, proc_info, wu_assignment in zip(selected_work_ids,
                                               selected_proc_info, solution):
      print('--------------------------')
      print('Work unit id: %d' % wu_id)
      print('--------------------------')
      for proc_id, proc_assignment in zip(proc_info[0], wu_assignment):
        for server, gpu in proc_assignment:

          server1 = selected_server_ids[server]
          assignment[wu_id][proc_id] = server1

          if proc_id in assignment[wu_id]:
            assert assignment[wu_id][proc_id] == server1
          else:
            assignment[wu_id][proc_id] = server1

          if proc_id in gpu_assignment[wu_id]:
            gpu_assignment[wu_id][proc_id].append(gpu)
            gpu_assignment[wu_id][proc_id].sort()
          else:
            gpu_assignment[wu_id][proc_id] = [gpu]

          print('Process %d assignment server: %d GPU: %d' %
                (proc_id, server1, gpu))

    return assignment, gpu_assignment

  def _schedule_cpu(self, scheduling_constraints, cpu_overload_obj_coeff,
                    cpu_load_balancing_obj_coeff,
                    cpu_wu_consolidation_obj_coeff):
    self._cpu_scheduler = LiaisonCPUScheduler(
        self.servers,
        scheduling_constraints,
        self.colocation_constraints,
        overload_obj_coeff=cpu_overload_obj_coeff,
        load_balancing_obj_coeff=cpu_load_balancing_obj_coeff,
        wu_consolidation_obj_coeff=cpu_wu_consolidation_obj_coeff)

    for wu in self.wunits:
      self._cpu_scheduler.add_work_unit(wu)

    assignment = self._cpu_scheduler.solve_cplex(
        None if self.time_limit is None else self.time_limit / 2)

    print('--------------------------')
    print('CPU assignment')
    for wu_id, wu_assignment in enumerate(assignment):
      print('--------------------------')
      print('Work unit id: %d' % wu_id)
      print('--------------------------')
      for proc_id, proc_assignment in enumerate(wu_assignment):
        print('Process %d assignment: %d' % (proc_id, proc_assignment))

    return assignment

  def get_assignment(self):
    return self._server_assignments, self._gpu_assignments
