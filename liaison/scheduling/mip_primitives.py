try:
  import cplex
except ModuleNotFoundError:
  pass


class Variable:

  def __init__(self,
               name,
               lower_bound=None,
               upper_bound=None,
               constraints=None):
    self.name = name
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound
    # add lower bound and upper bound to constraints if given.
    if constraints:
      assert isinstance(constraints, list)
      if lower_bound is not None:
        c = Constraint('GE', lower_bound)
        c.add_term(name, 1)
        constraints.append(c)
      if upper_bound is not None:
        c = Constraint('LE', upper_bound)
        c.add_term(name, 1)
        constraints.append(c)

  def convert_to_ortools_solver_format(self, solver):
    infinity = solver.infinity()
    lower_bound = self.lower_bound
    upper_bound = self.upper_bound
    if lower_bound is None:
      lower_bound = -infinity
    if upper_bound is None:
      upper_bound = infinity
    return solver.IntVar(lower_bound, upper_bound, self.name)

  def add_to_cplex_solver(self, solver):
    solver.variables.add(names=[self.name])
    solver.variables.set_types(self.name, solver.variables.type.integer)
    if self.lower_bound:
      solver.variables.set_lower_bounds(self.name, self.lower_bound)
    if self.upper_bound:
      solver.variables.set_upper_bounds(self.name, self.upper_bound)


class Process:

  def __init__(self, id, cpu_cost, mem_cost, gpu_compute_cost, gpu_mem_cost):
    self._id = id
    self.cpu_cost = cpu_cost
    self.mem_cost = mem_cost
    self.gpu_compute_cost = gpu_compute_cost
    self.gpu_mem_cost = gpu_mem_cost


class Constraint:

  def __init__(self, sense, rhs):
    """
      senses: "E", "LE", "GE"
    """
    self.sense = sense
    assert isinstance(rhs, float) or isinstance(rhs, int)
    self.rhs = rhs
    self.var_names = []
    self.coeffs = []

  def add_term(self, var_name, coeff):
    self.var_names.append(var_name)
    self.coeffs.append(coeff)

  def add_to_ortools_solver(self, solver, varnames2vars):
    infinity = solver.infinity()
    if self.sense == 'LE':
      ct = solver.Constraint(-infinity, self.rhs, '')
    elif self.sense == 'GE':
      ct = solver.Constraint(self.rhs, infinity, '')
    elif self.sense == 'E':
      ct = solver.Constraint(self.rhs, self.rhs, '')
    else:
      raise Exception('Unknown option %s' % self.sense)

    for var_name, coeff in zip(self.var_names, self.coeffs):
      ct.SetCoefficient(varnames2vars[var_name], coeff)
    return ct

  def add_to_cplex_solver(self, solver):
    sense = self.sense
    if sense == 'LE':
      sense = 'L'  # different terminology
    elif sense == 'GE':
      sense = 'G'

    solver.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=self.var_names, val=self.coeffs)],
        senses=[sense],
        rhs=[self.rhs])


class Objective:
  """Always minimize the objective."""

  def __init__(self):
    self.var_names = []
    self.coeffs = []

  def add_term(self, var_name, coeff):
    assert isinstance(var_name, str)
    self.var_names.append(var_name)
    self.coeffs.append(coeff)

  @staticmethod
  def combine_objectives(objs, coeffs):
    assert len(objs) == len(coeffs)
    combined_obj = Objective()
    for obj, obj_coeff in zip(objs, coeffs):
      for var_name, coeff in zip(obj.var_names, obj.coeffs):
        combined_obj.add_term(var_name, obj_coeff * coeff)
    return combined_obj

  def add_to_ortools_solver(self, solver, varnames2vars):
    objective = solver.Objective()
    objective.SetMinimization()
    for var_name, coeff in zip(self.var_names, self.coeffs):
      assert var_name in varnames2vars
      objective.SetCoefficient(varnames2vars[var_name], coeff)
    return objective

  def add_to_cplex_solver(self, solver):
    solver.objective.set_linear(zip(self.var_names, self.coeffs))
