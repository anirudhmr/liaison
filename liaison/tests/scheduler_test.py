from collections import namedtuple

from absl.testing import absltest
from liaison.scheduling import LiaisonScheduler

SERVER = namedtuple('Server', ['cpu', 'mem', 'gpu_compute', 'gpu_mem'])
PROCESS = namedtuple(
    'Process', ['cpu_cost', 'mem_cost', 'gpu_compute_cost', 'gpu_mem_cost'])


class SchedulingTest(absltest.TestCase):

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

  def testSolve(self):
    # First situation
    servers, work_units = self._setup()
    solver = LiaisonScheduler(servers,
                              overload_obj_coeff=1,
                              load_balancing_obj_coeff=1,
                              wu_consolidation_obj_coeff=8 + .1)
    for wu in work_units:
      solver.add_work_unit(wu)
    assignment = solver.solve()
    for wu_id, wu_assignment in enumerate(assignment):
      print('--------------------------')
      print('Work unit id: %d' % wu_id)
      print('--------------------------')
      for proc_id, proc_assignment in enumerate(wu_assignment):
        assert proc_assignment.count(1) == 1
        assignment = proc_assignment.index(1)
        print('Process %d assignment: %d' % (proc_id, assignment))


if __name__ == '__main__':
  absltest.main()
