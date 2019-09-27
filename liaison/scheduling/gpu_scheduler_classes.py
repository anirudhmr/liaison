from .mip_primitives import (Constraint, Expression, Objective, Variable,
                             MIPTracker, compute_min, compute_max,
                             compute_relu)
import copy


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

  def __init__(self, id, gpu_compute_cost, gpu_mem_cost):
    self.id = id
    self.gpu_compute_cost = gpu_compute_cost
    self.gpu_mem_cost = gpu_mem_cost
    self.reqs = []

  def send_requests(self, servers):
    """Sends allocation requests to the servers."""
    n_requests = len(self.gpu_compute_cost)  # number of gpus requesting
    responses = [[] for _ in range(n_requests)]
    reqs = [
        Request(cost, mem)
        for cost, mem in zip(self.gpu_compute_cost, self.gpu_mem_cost)
    ]

    for server in servers:
      resps = server.handle_bundled_requests(reqs, self.id)
      for req, response in zip(reqs, resps):
        req.add_response(response)

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
      self.procs.append(Process(i, spec.gpu_compute_cost, spec.gpu_mem_cost))

  def send_requests(self, servers, mip):
    for proc in self.procs:
      proc.send_requests(servers)
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

  def __init__(self, id, gpu_compute, gpu_mem, mip):
    assert isinstance(gpu_compute, list)
    assert isinstance(gpu_mem, list)
    self.id = id
    self.gpu_compute = gpu_compute
    self.gpu_mem = gpu_mem
    self.mip = mip

    self._gpu_resources = []
    for i, (c, m) in enumerate(zip(gpu_compute, gpu_mem)):
      self._gpu_resources.append(GPUResource(id, i, c, m, mip))
    self._request_id = 0

  def handle_bundled_requests(self, reqs, pid):
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
