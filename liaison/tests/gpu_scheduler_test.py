from collections import namedtuple

from absl.testing import absltest, parameterized
from liaison.scheduling import LiaisonGPUScheduler

SERVER = namedtuple('Server', ['gpu_compute', 'gpu_mem', 'mem'])
PROCESS = namedtuple('Process',
                     ['gpu_compute_cost', 'gpu_mem_cost', 'mem_cost'])


class SchedulingTest(parameterized.TestCase):

  def _setup(self):
    N_WORK_UNITS = 2
    N_PROCS = 2
    N_SERVERS = 2

    servers = [SERVER([8, 8, 8, 8], [32, 16, 8, 4], 16)] * N_SERVERS
    work_units = []
    for i in range(N_WORK_UNITS):
      procs = []
      for j in range(N_PROCS):
        proc = PROCESS([5, 5], [12, 8], 4)
        procs.append(proc)
      work_units.append(procs)

    return servers, work_units

  def _large_setup(self):
    N_WORK_UNITS = 4
    N_PROCS = 10
    N_SERVERS = 32

    servers = [SERVER([8] * 4, [32, 16, 8, 4], 16)] * N_SERVERS
    work_units = []
    for i in range(N_WORK_UNITS):
      procs = []
      for j in range(N_PROCS):
        proc = PROCESS([5, 5], [12, 8], 4)
        procs.append(proc)
      work_units.append(procs)

    return servers, work_units

  @parameterized.parameters((False, ), (True, ))
  def testSolve(self, large=False):
    if large:
      f = self._large_setup
    else:
      f = self._setup
    servers, work_units = f()
    solver = LiaisonGPUScheduler(servers,
                                 work_units, {}, [],
                                 overload_obj_coeff=1,
                                 load_balancing_obj_coeff=1,
                                 wu_consolidation_obj_coeff=10)
    assignment = solver.solve_cplex(time_limit=90)
    for wu_id, wu_assignment in enumerate(assignment):
      print('--------------------------')
      print('Work unit id: %d' % wu_id)
      print('--------------------------')
      for proc_id, proc_assignment in enumerate(wu_assignment):
        for server, gpu in proc_assignment:
          print('Process %d assignment server: %d GPU: %d' %
                (proc_id, server, gpu))


if __name__ == '__main__':
  absltest.main()
