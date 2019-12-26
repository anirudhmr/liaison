import copy

from .mip_primitives import (Constraint, Expression, MIPTracker, Objective,
                             Variable, compute_max, compute_min, compute_relu)


class ResponseVariable:
  """
    Records request of an allocation of a single gpu from a server.
  """

  def __init__(self, name, mip):
    self.name = name + '/request_fulfilled_var'
    self.var = mip.new_variable(
        self.name,
        0,
        1,
    )
    self.expr = self.var.to_expression()


class Response:
  """
    Combine multiple requests together.
  """

  def __init__(self, name, response_vars):
    self.name = name
    self.response_vars = response_vars
    self.expr = Expression.sum_expressions(
        [response_var.expr for response_var in response_vars])


class Request:

  def __init__(self, gpu_compute, gpu_mem):
    self.gpu_compute = gpu_compute
    self.gpu_mem = gpu_mem
    self.responses = []

  def add_response(self, response):
    """Receive a response for this request from a server."""
    self.responses.append(response)


class Process:
  """
   Process creates requests to server for GPUs.
  """

  def __init__(self, id, wid, gpu_compute_cost, gpu_mem_cost, mem):
    self.id = id
    self.wid = wid
    self.gpu_compute_cost = gpu_compute_cost
    self.gpu_mem_cost = gpu_mem_cost
    self.mem = mem
    self.reqs = []

  def __str__(self):
    ret = f'Process id: {self.id}\n'
    ret += f'gpu_mem_cost: {self.gpu_mem_cost}, gpu_compute_cost: {self.gpu_compute_cost}\n'
    return ret

  def send_requests(self, servers, mip, constraint=None):
    """
       Sends allocation requests to the servers.
       constraints -> List[int] (server_ids where the process could go)
    """
    n_requests = len(self.gpu_compute_cost)  # number of gpus requesting
    responses = [[] for _ in range(n_requests)]
    reqs = [
        Request(cost, mem)
        for cost, mem in zip(self.gpu_compute_cost, self.gpu_mem_cost)
    ]

    if constraint:
      c = mip.new_constraint('E',
                             1,
                             name='pl_constraint_wid_%d_pid_%d' %
                             (self.wid, self.id))

    if reqs:
      for server in servers:
        resps = server.handle_bundled_requests(reqs, self.mem, self.wid,
                                               self.id)
        for req, response in zip(reqs, resps):
          req.add_response(response)

        # check if constraint requires to set the process to this server.
        if constraint:
          if server.id in constraint:
            c.add_expression(
                Expression.sum_expressions(
                    [resp.expr for resp in req.responses]))

      self.reqs.extend(reqs)

  def add_fulfillment_constraints(self, mip):
    """Each request must be satisfied by exactly one server."""
    for req in self.reqs:
      # sum of all server responses for this request = 1
      e = Expression.sum_expressions([resp.expr for resp in req.responses])
      c = e.to_constraint('E', 1)
      mip.add_constraint(c)

  def get_placement_indicators(self):
    """
      Returns List[exprs] whose len == len(servers).
    """
    # Note that using `handle_bundled_requests` means
    # that all the responses for all the requests will
    # be same due to bundled request colocation constraint
    # so we simply pick the first request.
    # Also, note that sum of responses will not be > 1
    # due to fulfillment constraint.
    return [resp.expr.copy() for resp in self.reqs[0].responses]


class WorkUnit:

  def __init__(self, id, proc_specs):
    self.procs = []
    self.id = id
    for i, spec in enumerate(proc_specs):
      self.procs.append(
          Process(i, id, spec.gpu_compute_cost, spec.gpu_mem_cost,
                  spec.mem_cost))

  def send_requests(self, servers, mip, wu_sched_constraints=None):
    """
      wu_sched_constraints -> Dict[pid] -> List[server_ids]
                              pid should be scheduled on one of server_ids
    """
    wu_sched_constraints = {} if wu_sched_constraints is None else wu_sched_constraints
    for proc in self.procs:
      responses = proc.send_requests(servers, mip,
                                     wu_sched_constraints.get(proc.id, None))
      proc.add_fulfillment_constraints(mip)

  def get_wu_consolidation_obj(self, mip):
    """
      Place processes from the same work unit on the same server.
    """
    # server_responses[i] = sum of responses from ith server of all the processes in this wu
    server_responses = []
    for proc in self.procs:
      server_indicators = proc.get_placement_indicators()
      if server_responses:
        for i, (resp, indicator) in enumerate(
            zip(server_responses, server_indicators)):
          server_responses[i] = Expression.sum_expressions([resp, indicator])
      else:
        server_responses = server_indicators

    obj = mip.new_objective('wu_consolidation_%d' % self.id)
    for i, resp in enumerate(server_responses):
      v = compute_min([resp, Expression(1)],
                      'wu_consolidation_%d/server_%d' % (self.id, i), [0, 1],
                      [len(self.procs), 1], mip)
      obj.add_term(v.name, 1)

    return obj

  def get_assignment_vars(self, solver):
    """Returns
    List[List[Tuple[server_id, gpu_id]]]
    """
    ass_vars = []
    for proc in self.procs:
      proc_ass_vars = []
      for req in proc.reqs:
        gpu = None
        server_i = None
        for i, resp in enumerate(req.responses):
          ass = solver.solution.get_values(
              [var.name for var in resp.response_vars])
          ass = list(map(int, ass))
          if 1 in ass:
            assert ass.count(1) == 1
            assert gpu is None
            gpu = ass.index(1)
            server_i = i
        assert gpu is not None
        proc_ass_vars.append((server_i, gpu))
      ass_vars.append(proc_ass_vars)
    return ass_vars


class GPUResource:

  def __init__(self, server_id, id, gpu_compute, gpu_mem, mip):
    self.id = id
    self.server_id = server_id
    self.gpu_compute = gpu_compute
    self.gpu_mem = gpu_mem
    self.mip = mip
    self._relu_load = None
    self.max_load = 0
    self.load_expr = Expression(-self.gpu_compute)
    self.mem_constraint = mip.new_constraint(
        'LE', self.gpu_mem,
        'server_%d_gpu_res_%d_mem_constraint' % (server_id, id))

  def handle_request(self, req, pid, req_id):
    resp = ResponseVariable(
        'server_%d_gpu_res_%d_pid_%d_req_%d' %
        (self.server_id, self.id, pid, req_id), self.mip)
    e = resp.expr.copy()
    e.scale_by(req.gpu_compute)
    self.load_expr = Expression.sum_expressions([self.load_expr, e])

    e = resp.expr.copy()
    e.scale_by(req.gpu_mem)
    self.mem_constraint.add_expression(e)
    self.max_load += req.gpu_compute
    return resp

  def compute_relu_load(self):
    name = 'server_%d_gpu_res_%d_load_relu_var' % (self.server_id, self.id)
    return compute_relu(self.load_expr, name, -self.gpu_compute, self.max_load,
                        self.mip)

  def get_relu_load(self):
    # cache it to avoid duplicate var and constraint creation.
    if self._relu_load is None:
      self._relu_load = self.compute_relu_load()
    return self._relu_load


class Server:

  def __init__(self,
               id,
               gpu_compute,
               gpu_mem,
               mem,
               mip,
               colocation_constraints=None):
    """
      colocation_constraints -> List[List[Tuple[wid, pid]]]
    """
    assert isinstance(gpu_compute, list)
    assert isinstance(gpu_mem, list)
    self.id = id
    self.gpu_compute = gpu_compute
    self.gpu_mem = gpu_mem
    self.mem = mem
    self.mip = mip
    self._mem_constraint = mip.new_constraint('LE', mem,
                                              'server_%d_mem_constraint' % id)
    self._colocation_constraints = [] if colocation_constraints is None else colocation_constraints

    self._gpu_resources = []
    for i, (c, m) in enumerate(zip(gpu_compute, gpu_mem)):
      self._gpu_resources.append(GPUResource(id, i, c, m, mip))
    self._request_id = 0

    # [wid, pid] -> expr (indicating if (wid, pid) gets assigned to this server)
    self._expr_for_colocation_constraints = dict()

  def __str__(self):
    ret = f'Server id: {self.id}\n'
    if self.gpu_mem:
      ret += f'gpu_mem: {self.gpu_mem}. gpu_compute: {self.gpu_compute}'
    return ret

  def handle_bundled_requests(self, reqs, proc_mem, wid, pid):
    combined_responses = []
    for i, req in enumerate(reqs):
      responses = [
          res.handle_request(req, pid, self._request_id + i)
          for res in self._gpu_resources
      ]

      combined_responses.append(
          Response(
              'server_%d_pid_%d_req_%d_combined_response' % (
                  self.id,
                  pid,
                  i + self._request_id,
              ), responses))

    # First add colocation constraints.
    # All requests must either be satisfied by this server or not.
    for resp in combined_responses[1:]:
      c = Expression.diff_expressions(combined_responses[0].expr,
                                      resp.expr).to_constraint('E', 0)
      self.mip.add_constraint(c)

    # Now add orthogonality constraints:
    # Two requests should not recieve same GPU Resource
    # constraints[i] stands for total assignments to GPUResource[i]
    # in this request bundle should be <= 1
    constraints = [
        self.mip.new_constraint('LE', 1, name=None)
        for _ in range(len(self._gpu_resources))
    ]
    for resp in combined_responses:
      assert len(resp.response_vars) == len(constraints)
      for i, var in enumerate(resp.response_vars):
        constraints[i].add_term(var.name, 1)

    # Now add server mem constraint:
    # Take the first  response
    # If it's fulfilled then due to colocation constraint all the
    # other requests must also be satisfied.
    # If the first response is an accept then add the memory of the process
    # to the constraint
    # Also response is bounded between 0 and 1.
    resp = combined_responses[0].expr.copy()
    resp.scale_by(proc_mem)
    self._mem_constraint.add_expression(resp)

    # Now handle the colocation constraints
    # For this we keep track of few expressions for every (wid, pid) pair
    # _expr_for_colocation_constraints[(wid, pid)] = E
    # E is an expressions indicating if (wid, pid) process is allocated
    # on this server.

    e1 = self._expr_for_colocation_constraints[(
        wid, pid)] = combined_responses[0].expr.copy()

    for constraint in self._colocation_constraints:
      # check if valid for this process
      if (wid, pid) in constraint:
        for coloc_wid, coloc_pid in constraint:
          if (wid, pid) != (coloc_wid, coloc_pid):
            if (coloc_wid, coloc_pid) in self._expr_for_colocation_constraints:
              e2 = self._expr_for_colocation_constraints[(coloc_wid,
                                                          coloc_pid)]
              # add constraint e1 - e2 = 0
              c = self.mip.new_constraint('E', 0, name=None)
              c.add_expression(Expression.diff_expressions(e1, e2))

    self._request_id += len(reqs)
    return combined_responses

  def get_utilization_obj(self):
    obj = self.mip.new_objective('server_%d_utilization_obj' % self.id)
    for i, res in enumerate(self._gpu_resources):
      obj.add_expression(res.get_relu_load().to_expression())
    return obj

  def get_load_balancing_obj(self):
    obj = self.mip.new_objective('server_%d_load_balancing_obj' % self.id)
    loads = []
    for i, res in enumerate(self._gpu_resources):
      loads += [res.get_relu_load().to_expression()]
    var_name = 'server_%d_load_balancing_max_var' % self.id
    v = compute_max(loads, var_name,
                    [-res.gpu_compute for res in self._gpu_resources],
                    [res.max_load for res in self._gpu_resources], self.mip)
    var_name = 'server_%d_load_balancing_max_var' % self.id
    obj.add_term(v.name, 1)
    return obj
