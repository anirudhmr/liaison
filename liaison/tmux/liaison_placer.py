import re
from multiprocessing.pool import ThreadPool

from ccc.src import NodeLoader, SlurmNode
from liaison.scheduling import ScheduleManager

RES_DIRS = [
    '/'.join(__file__.split('/')[:-2]),
]


class LiaisonPlacer:

  def __init__(self,
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

    nodes, whitelist_nodes = self._make_nodes(cluster_config,
                                              filter_nodes_regex,
                                              whitelist_nodes)
    wunits = self._get_wunits(exps)
    # use whitelist_nodes only for hard placements.
    filtered_wunits = self._filter_out_hard_placements(wunits,
                                                       nodes + whitelist_nodes)
    filtered_wunits, slurm_nodes, nodes = self._filter_out_slurm_placements(
        filtered_wunits, nodes, pl_constraints)

    if len(filtered_wunits) == 0:
      return

    if len(nodes) == 0:
      if len(slurm_nodes) == 1:
        for procs in filtered_wunits:
          for proc in procs:
            self._set_placement(proc, slurm_nodes[0])
      else:
        raise Exception(
            'More than one slurm nodes remaining to assign to pending processes.'
        )
      return

    with ThreadPool(len(nodes)) as pool:
      pool.map(lambda node: node.collect_spy_stats(spy_measurement_interval),
               nodes)

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
          coloc_constraints.append(l2)
    del coloc_constraints

    manager = ScheduleManager(nodes, wunits, sched_constraints,
                              coloc_constraints2, **config)

    ass, gpu_ass = manager.get_assignment()
    for wu_assignment, wu_gpu_assignment, procs in zip(ass, gpu_ass, wunits):
      for proc_id, (proc, server) in enumerate(zip(procs, wu_assignment)):
        self._set_placement(proc, nodes[server])
        if proc_id in wu_gpu_assignment:
          proc.set_gpus(wu_gpu_assignment[proc_id])

  def _make_nodes(self, cluster_config, filter_nodes_regex,
                  whitelist_node_list):
    nodes = NodeLoader(cluster_config, filter_nodes_regex).nodes
    print('Filtered nodes (%d): ' % len(nodes))
    for node in nodes:
      print(node.name)

    whitelist_nodes = NodeLoader(cluster_config,
                                 '|'.join(whitelist_node_list)).nodes
    print('Whitelisted nodes (%d): ' % len(whitelist_nodes))
    for node in whitelist_nodes:
      print(node.name)

    with ThreadPool(len(nodes) + len(whitelist_nodes)) as pool:
      pool.map(lambda node: node.setup(res_dirs=RES_DIRS),
               nodes + whitelist_nodes)
    return nodes, whitelist_nodes

  def _set_placement(self, proc, node):
    allocation = node.allocate(cpu=proc.cpu_cost,
                               mem=proc.mem_cost,
                               n_gpus=len(proc.gpu_mem_cost))
    proc.set_allocation(allocation)
    proc.set_placement(node)

  def _get_wunits(self, exps):
    return [[
        proc for pg in exp.list_process_groups()
        for proc in pg.list_processes()
    ] + [proc for proc in exp.list_processes()] for exp in exps]

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

  def _filter_out_slurm_placements(self, wunits, nodes, pl_constraints):
    """Filter out the slurm nodes and the procs that go on to slurm nodes."""
    slurm_nodes = []
    filtered_nodes = []
    for node in nodes:
      if isinstance(node, SlurmNode):
        slurm_nodes.append(node)
      else:
        filtered_nodes.append(node)

    filtered_wunits = []
    for procs in wunits:
      filtered_wunit = []
      for proc in procs:
        # If proc matched already with slurm node in a previous pl constraint.
        proc_matched = False
        for proc_regex, node_regex in pl_constraints:
          if re.search(proc_regex, proc.name):
            matches = []
            # Find the slurm nodes that match the constraint
            for node in slurm_nodes:
              if re.search(node_regex, node.name):
                matches.append(node)

            # if len(matches) == 0 then the constraint is irrelevant
            # for slurm nodes and for this particular proc.

            if len(matches) > 1:
              # Not clear which slurm node should be used for this proc
              # raise an exception to avoid this input.
              raise Exception('Following slurm nodes match for proc %s: %s' %
                              (proc.name, ' '.join(matches)))

            elif len(matches) == 1:
              # if process already matched another constraint then
              # raise exception.
              # TODO: It's okay when the previous constraint also matched to the
              # same slurm node. Don't raise exception in that case.
              if proc_matched:
                raise Exception(
                    'Process %s matches multiple slurm node placement constraints'
                    % proc.name)
              else:
                proc_matched = True
                self._set_placement(proc, matches[0])

        if proc_matched:
          # dont include it in filtered procs
          pass
        else:
          filtered_wunit.append(proc)

      if filtered_wunit:
        filtered_wunits.append(filtered_wunit)

    return filtered_wunits, slurm_nodes, filtered_nodes
