import pdb
import re
from multiprocessing.pool import ThreadPool

import numpy as np
from ccc.src import LSFNode, NodeLoader, SlurmNode
from liaison.scheduling import ScheduleManager
from liaison.utils import ConfigDict

RES_DIRS = [
    '/'.join(__file__.split('/')[:-2]),  # liaison directory
    '/'.join(__file__.split('/')[:-3]) + '/.git',  # .git directory
]


class LiaisonPlacer:

  def __init__(self,
               xid,
               exps,
               cluster_config,
               filter_nodes_regex,
               whitelist_nodes,
               spy_measurement_interval,
               pl_constraints=None,
               coloc_constraints=None,
               **config):
    """
      config will be passed to the schedule manager.
      exps: List[symphony.Experiment]
      cluster_config: Parsed namespace that can be passed to nodeloader
      filter_nodes_regex: Used by nodeloader to filter nodes
      whitelist_nodes: whitelisted set of nodes not used for default placement
      pl_constraints: List[Tuple[str, str]]
              str1 regex of proc names
              str2 regex of node names
      coloc_constraints: List[List[str]]
    """
    self.config = ConfigDict(config)
    self.xid = xid
    nodes, whitelist_nodes = self._make_nodes(cluster_config, filter_nodes_regex, whitelist_nodes)
    wunits = self._get_wunits(exps)
    # use whitelist_nodes only for hard placements.
    filtered_wunits = self._filter_out_hard_placements(wunits, nodes + whitelist_nodes)
    filtered_wunits, slurm_or_lsf_nodes, filtered_nodes = self._filter_out_slurm_or_lsf_placements(
        filtered_wunits, nodes, pl_constraints)

    if len(filtered_wunits) == 0:
      return

    if len(filtered_nodes) == 0:
      # if no other node left but few procs still remaining
      # they are to be scheduled on the slurm node.
      if len(slurm_or_lsf_nodes) == 1:
        for procs in filtered_wunits:
          for proc in procs:
            self._set_placement(proc, slurm_or_lsf_nodes[0])
      else:
        raise Exception('More than one slurm nodes remaining to assign to pending processes.')
      return

    nodes = filtered_nodes
    with ThreadPool(len(nodes)) as pool:
      pool.map(lambda node: node.collect_spy_stats(spy_measurement_interval), nodes)

    wunits = filtered_wunits
    # convert pl_constraints to the following format:
    # Dict[(wid, pid)] => List[server_ids]
    sched_constraints = {}
    for proc_regex, node_regex in pl_constraints:
      for sid, node in enumerate(nodes):
        # If the node is relevant
        if re.search(node_regex, node.name):
          for wid, procs in enumerate(wunits):
            for pid, proc in enumerate(procs):
              # this process is relevant
              if re.search(proc_regex, proc.name):
                if (wid, pid) in sched_constraints:
                  sched_constraints[(wid, pid)].append(sid)
                else:
                  sched_constraints[(wid, pid)] = [sid]

    # convert coloc_constraints to the following format:
    # List[List[Tuple[wid, pid]]]
    coloc_constraints2 = []

    # for each workunit
    for wid, procs in enumerate(wunits):
      # translate the coloc constraints of procs in
      # this work unit to pids
      for l in coloc_constraints:
        l2 = []
        for pid, proc in enumerate(procs):
          for regex in l:
            if re.search(regex, proc.name):
              l2.append((wid, pid))
              break
        if l2:
          coloc_constraints2.append(l2)
    del coloc_constraints

    manager = ScheduleManager(nodes, wunits, sched_constraints, coloc_constraints2, **config)

    ass, gpu_ass = manager.get_assignment()
    for wu_assignment, wu_gpu_assignment, procs in zip(ass, gpu_ass, wunits):
      for proc_id, (proc, server) in enumerate(zip(procs, wu_assignment)):
        self._set_placement(proc, nodes[server])
        if proc_id in wu_gpu_assignment:
          proc.set_gpus(wu_gpu_assignment[proc_id])

  def _make_nodes(self, cluster_config, filter_nodes_regex, whitelist_node_list):
    nodes = NodeLoader(cluster_config, filter_nodes_regex).nodes
    print('Filtered nodes (%d): ' % len(nodes))
    for node in nodes:
      print(node.name)

    whitelist_nodes = NodeLoader(cluster_config, '|'.join(whitelist_node_list)).nodes
    print('Whitelisted nodes (%d): ' % len(whitelist_nodes))
    for node in whitelist_nodes:
      print(node.name)

    with ThreadPool(len(nodes) + len(whitelist_nodes)) as pool:
      pool.map(lambda node: node.setup(res_dirs=RES_DIRS), nodes + whitelist_nodes)
    return nodes, whitelist_nodes

  def _set_placement(self, proc, node):
    allocation = node.allocate(cpu=proc.cpu_cost,
                               mem=proc.mem_cost,
                               n_gpus=len(proc.gpu_mem_cost),
                               name=f'{self.xid}_{proc.name}')
    proc.set_allocation(allocation)
    proc.set_placement(node)

  def _slurm_wunit_based_allocation(self, wunit_matches):
    # colocate all procs of a work-unit on a *single* slurm
    # allocation (single slurm worker node)
    # determine the # of gpus needed.
    procs, nodes = list(zip(*wunit_matches))
    if len(procs) == 0:
      return
    # assert only one node selected
    assert len(set(nodes)) == 1
    node = nodes[0]

    exclusive_procs = list(filter(lambda p: p.slurm_exclusive_gpu, procs))
    non_exclusive_procs = list(filter(lambda p: not p.slurm_exclusive_gpu, procs))

    if self.config.slurm_colocate_wunit:
      # all the non-exclusive procs share gpus
      # gpus for the exclusive gpu procs.
      n_gpus = sum(len(p.gpu_mem_cost) for p in exclusive_procs)
      # gpus for all the non-exclusive-procs
      n_gpus += max(len(p.gpu_mem_cost) for p in non_exclusive_procs)
      # colocate an entire workunit on a single slurm node.
      allocation = node.allocate(cpu=sum([p.cpu_cost for p in procs]),
                                 mem=sum([p.mem_cost for p in procs]),
                                 n_gpus=n_gpus,
                                 name=f'{self.xid}')
      for proc in procs:
        proc.set_allocation(allocation)

      # assign gpus to procs.
      gpu_id = 0
      for proc in exclusive_procs:
        n_gpus = len(proc.gpu_mem_cost)
        proc.set_gpus(allocation.gpu_visible_ids[gpu_id:gpu_id + n_gpus])
        gpu_id += n_gpus

      for proc in non_exclusive_procs:
        proc.set_gpus(allocation.gpu_visible_ids[gpu_id:gpu_id + len(proc.gpu_mem_cost)])

    elif self.config.slurm_per_gpu_allocation:
      # create one slurm allocation for each exclusive gpu
      # all other non-exclusive procs share a *single* allocation
      # all the other non-gpu procs assigned an arbitrary allocation (no new allocation created).
      allocation = None
      for proc in exclusive_procs:
        allocation = node.allocate(cpu=proc.cpu_cost,
                                   mem=proc.mem_cost,
                                   n_gpus=len(proc.gpu_mem_cost),
                                   name=f'{self.xid}_{proc.name}')
        proc.set_allocation(allocation)
        proc.set_gpus(allocation.gpu_visible_ids)
      # n_gpus required for non_exclusive allocation
      n_gpus = max([len(p.gpu_mem_cost) for p in non_exclusive_procs])
      if allocation is None or n_gpus > 0:
        allocation = node.allocate(cpu=sum([p.cpu_cost for p in non_exclusive_procs]),
                                   mem=sum([p.mem_cost for p in non_exclusive_procs]),
                                   n_gpus=n_gpus,
                                   name=f'{self.xid}_non_exclusive_allocation')
      for proc in non_exclusive_procs:
        proc.set_allocation(allocation)
        proc.set_gpus(allocation.gpu_visible_ids[:len(proc.gpu_mem_cost)])

    else:
      raise Exception('Unknown case encountered.')

    for proc in procs:
      proc.set_placement(node)

  def _get_wunits(self, exps):
    return [[proc for pg in exp.list_process_groups()
             for proc in pg.list_processes()] + [proc for proc in exp.list_processes()]
            for exp in exps]

  def _filter_out_hard_placements(self, wunits, nodes):
    """Filter out the procs that have a hard placement to one of the nodes."""
    filtered_wunits = []
    for procs in wunits:
      filtered_wunit = []
      for proc in procs:
        if proc.hard_placement:
          try:
            i = [node.name for node in nodes].index(proc.hard_placement)
            self._set_placement(proc, nodes[i])
          except ValueError as e:
            print('Unable to find %s in nodelist' % proc.hard_placement)
            raise e
        else:
          filtered_wunit.append(proc)

      if filtered_wunit:
        filtered_wunits.append(filtered_wunit)

    return filtered_wunits

  def _set_gpus_for_slurm_or_lsf_node(self, proc, node):
    assert isinstance(node, (LSFNode, SlurmNode))

    if isinstance(node, SlurmNode):
      try:
        s = node.exec_commands("bash -c 'env | grep CUDA'", allocation=proc.allocation)
      except Exception:
        s = ''

      s = s.replace('\n', '').replace('\r', '')
      if len(s):
        # ex: CUDA_VISIBLE_DEVICES=2,3
        s = s.split('CUDA_VISIBLE_DEVICES=')[-1].split(',')
      else:
        # no gpus to set.
        return
    else:
      s = list(map(str, range(proc.allocation.ngpus)))

    return proc.set_gpus(s)

  def _filter_out_slurm_or_lsf_placements(self, wunits, all_nodes, pl_constraints):
    """Filter out the slurm, lsf nodes and the procs that go on to these nodes."""
    # slurm or lsf nodes
    nodes = []
    filtered_nodes = []
    for node in all_nodes:
      if isinstance(node, (LSFNode, SlurmNode)):
        nodes.append(node)
      else:
        filtered_nodes.append(node)

    filtered_wunits = []
    for procs in wunits:
      slurm_or_lsf_matches = []
      filtered_wunit = []
      for proc in procs:
        # If proc matched already with slurm node in a previous pl constraint.
        proc_matched = None
        for proc_regex, node_regex in pl_constraints:
          if re.search(proc_regex, proc.name):
            matches = []
            # Find the nodes that match the constraint
            for node in nodes:
              if re.search(node_regex, node.name):
                matches.append(node)

            if len(matches) == 0:
              # The constraint is irrelevant
              # for slurm nodes and for this particular proc.
              continue
            elif len(matches) == 1:
              # if process already matched another constraint then
              # raise Exception.
              # TODO: It's okay when the previous constraint also matched to the
              # same slurm node. Don't raise exception in that case.
              if proc_matched:
                raise Exception('Process %s matches multiple slurm node placement constraints' %
                                proc.name)
              else:
                proc_matched = matches[0]
            elif len(matches) > 1:
              # Not clear which node should be used for this proc
              # raise an exception to avoid this input.
              raise Exception('Following nodes match for proc %s: %s' %
                              (proc.name, ' '.join(matches)))
        if proc_matched:
          # dont include it in filtered procs
          slurm_or_lsf_matches.append((proc, matches[0]))
        else:
          filtered_wunit.append(proc)

      if self.config.slurm_colocate_wunit or self.config.slurm_per_gpu_allocation:
        # process wunit at a time.
        self._slurm_wunit_based_allocation(slurm_or_lsf_matches)
      else:
        for proc, match in slurm_or_lsf_matches:
          self._set_placement(proc, match)
          self._set_gpus_for_slurm_or_lsf_node(proc, match)
      filtered_wunits.append(filtered_wunit)

    # remove empty wunits.
    filtered_wunits = list(filter(lambda l: len(l) > 0, filtered_wunits))
    return filtered_wunits, nodes, filtered_nodes
