from collections import namedtuple

from absl.testing import absltest, parameterized
from liaison.scheduling import LiaisonCPUScheduler

SERVER = namedtuple('Server', ['cpu', 'mem', 'gpu_compute', 'gpu_mem'])
PROCESS = namedtuple(
    'Process', ['cpu_cost', 'mem_cost', 'gpu_compute_cost', 'gpu_mem_cost'])


class SchedulingTest(parameterized.TestCase):

  def _setup(self):
    N_WORK_UNITS = 3
    N_PROCS = 2
    N_SERVERS = 2

    servers = [SERVER(8, 64, None, None)] * N_SERVERS
    work_units = []
    for i in range(N_WORK_UNITS):
      procs = []
      for j in range(N_PROCS):
        proc = PROCESS(4, 0, None, None)
        procs.append(proc)
      work_units.append(procs)

    return servers, work_units

  def _setup_memory(self):
    N_WORK_UNITS = 2
    N_PROCS = 6
    N_SERVERS = 12

    servers = [SERVER(8, 16, None, None)] * N_SERVERS
    work_units = []
    for i in range(N_WORK_UNITS):
      procs = []
      for j in range(N_PROCS):
        proc = PROCESS(4, 6, None, None)
        procs.append(proc)
      work_units.append(procs)

    return servers, work_units

  def _large_setup(self):
    N_WORK_UNITS = 4
    N_PROCS = 10
    N_SERVERS = 32

    servers = [SERVER(8, 64, None, None)] * N_SERVERS
    work_units = []
    for i in range(N_WORK_UNITS):
      procs = []
      for j in range(N_PROCS):
        proc = PROCESS(4, 0, None, None)
        procs.append(proc)
      work_units.append(procs)

    return servers, work_units

  @parameterized.parameters((True, False), (True, True))
  def testSolve(self, memory=False, large=False):
    # First situation
    if memory:
      f = self._setup_memory
    elif large:
      f = self._large_setup
    else:
      f = self._setup
    servers, work_units = f()
    solver = LiaisonCPUScheduler(servers,
                                 overload_obj_coeff=1,
                                 load_balancing_obj_coeff=1,
                                 wu_consolidation_obj_coeff=10)
    for wu in work_units:
      solver.add_work_unit(wu)
    assignment = solver.solve_cplex(time_limit=90)
    for wu_id, wu_assignment in enumerate(assignment):
      print('--------------------------')
      print('Work unit id: %d' % wu_id)
      print('--------------------------')
      for proc_id, proc_assignment in enumerate(wu_assignment):
        print('Process %d assignment: %d' % (proc_id, proc_assignment))


if __name__ == '__main__':
  absltest.main()
