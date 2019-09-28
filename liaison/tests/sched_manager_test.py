from collections import namedtuple

from absl.testing import absltest, parameterized
from liaison.scheduling import ScheduleManager

Process = namedtuple(
    'Process',
    ['name', 'cpu_cost', 'mem_cost', 'gpu_compute_cost', 'gpu_mem_cost'])


class Node:

  def __init__(self, name, avail_cpu, avail_mem, avail_gpu_mem,
               avail_gpu_compute):
    self.name = name
    self.avail_cpu = lambda: avail_cpu
    self.avail_mem = lambda: avail_mem
    self.avail_gpu_mem = lambda: gpu_mem
    self.avail_gpu_compute = lambda scale: [
        scale['K80'] * c for c in gpu_compute
    ]


class Experiment:

  def __init__(self):
    self.procs = []

  def append(self, proc):
    self.procs.append(proc)

  def list_process_groups(self):
    return []

  def list_processes(self):
    return self.procs


class SchedulingTest(parameterized.TestCase):

  def _setup(self):
    N_WORK_UNITS = 2
    N_PROCS = 2
    N_SERVERS = 2

    servers = [
        Node(str(i), 8, 8, [4, 4, 4, 4], [32, 16, 8, 4])
        for i in range(N_SERVERS)
    ]
    work_units = []
    for i in range(N_WORK_UNITS):
      ex = Experiment()
      for j in range(N_PROCS):
        proc = Process('%d/%d' % (i, j), 4, 4, [1.5, 1.5], [12, 8])
        ex.append(proc)
      work_units.append(ex)

    return servers, work_units

  def _large_setup(self):
    N_WORK_UNITS = 4
    N_PROCS = 10
    N_SERVERS = 32

    servers = [
        Node(str(i), 8, 8, [4, 4, 4, 4], [32, 16, 8, 4])
        for i in range(N_SERVERS)
    ]
    work_units = []
    for i in range(N_WORK_UNITS):
      ex = Experiment()
      for j in range(N_PROCS):
        proc = Process('%d/%d' % (i, j), 4, 4, [1.5, 1.5], [12, 8])
        ex.append(proc)
      work_units.append(ex)

    return servers, work_units

  @parameterized.parameters((False, ), (True, ))
  def testSolve(self, large=False):
    if large:
      f = self._large_setup
    else:
      f = self._setup
    servers, work_units = f()
    manager = ScheduleManager(servers,
                              work_units,
                              scheduling_constraints={(0, 0): 0},
                              colocation_constraints=[[(0, 0), (0, 1)]],
                              time_limit=40)
    assignment, gpu_ass = manager.get_assignment()

    print('--------------------------')
    print('Combined placement')
    print('--------------------------')
    for wu_id, (wu_assignment,
                wu_gpu_assignment) in enumerate(zip(assignment, gpu_ass)):
      print('--------------------------')
      print('Work unit id: %d' % wu_id)
      print('--------------------------')
      for proc_id, server in enumerate(wu_assignment):
        print('Process %d assignment server: %d' % (proc_id, server), end='')
        if proc_id in wu_gpu_assignment:
          print(' GPU: %s' % (','.join(map(str, wu_gpu_assignment[proc_id]))))
        else:
          print()


if __name__ == '__main__':
  absltest.main()
