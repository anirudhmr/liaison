import copy
try:
  import cplex
except ModuleNotFoundError:
  pass


class Variable:

  def __init__(self, name, lower_bound=None, upper_bound=None):
    self.name = name
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound

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

  def to_expression(self):
    e = Expression()
    e.add_term(self.name, 1)
    return e


class Constraint:

  def __init__(self, sense, rhs, name=None):
    """
      senses: "E", "LE", "GE"
    """
    self.sense = sense
    assert isinstance(rhs, float) or isinstance(rhs, int)
    self.rhs = rhs
    self.var_names = []
    self.coeffs = []

  def add_term(self, var_name, coeff):
    assert isinstance(var_name, str)
    self.var_names.append(var_name)
    self.coeffs.append(coeff)

  def add_terms(self, var_names, coeffs):
    assert isinstance(var_names[0], str)
    self.var_names.extend(var_names)
    self.coeffs.extend(coeffs)

  def add_expression(self, expr):
    assert expr.constant == 0
    self.var_names.extend(expr.var_names)
    self.coeffs.extend(expr.coeffs)

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


class Expression:

  def __init__(self, constant=0):
    self.var_names = []
    self.coeffs = []
    self.constant = constant

  def add_constant(self, constant):
    self.constant += constant

  def add_term(self, var_name, coeff):
    self.var_names.append(var_name)
    self.coeffs.append(coeff)

  def add_terms(self, var_names, coeffs):
    self.var_names.extend(var_names)
    self.coeffs.extend(coeffs)

  def scale_by(self, const):
    self.coeffs = [coeff * const for coeff in self.coeffs]
    self.constant *= const

  def to_constraint(self, sense, rhs):
    c = Constraint(sense, rhs - self.constant)
    for var_name, coeff in zip(self.var_names, self.coeffs):
      c.add_term(var_name, coeff)
    return c

  @staticmethod
  def sum_expressions(exprs):
    e = Expression()
    for expr in exprs:
      e.add_constant(expr.constant)
      e.add_terms(expr.var_names, expr.coeffs)
    return e

  @staticmethod
  def diff_expressions(e1, e2):
    e = e1.copy()
    e.add_constant(-e2.constant)
    e.add_terms(e2.var_names, list(map(lambda k: -k, e2.coeffs)))
    return e

  def copy(self):
    return copy.deepcopy(self)


class Objective:
  """Always minimize the objective."""

  def __init__(self, name=None):
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

  def add_expression(self, expr):
    # ignore expression constant.
    self.var_names.extend(expr.var_names)
    self.coeffs.extend(expr.coeffs)

  def add_to_ortools_solver(self, solver, varnames2vars):
    objective = solver.Objective()
    objective.SetMinimization()
    for var_name, coeff in zip(self.var_names, self.coeffs):
      assert var_name in varnames2vars
      objective.SetCoefficient(varnames2vars[var_name], coeff)
    return objective

  def add_to_cplex_solver(self, solver):
    solver.objective.set_linear(zip(self.var_names, self.coeffs))


class MIPTracker:
  """
    Keeps track of created variables and constraints and objectives.
  """

  def __init__(self):
    self.varnames2var = dict()
    self.constraints = []
    self.objectives = []

  def new_variable(self, var_name, *args, **kwargs):
    v = Variable(var_name, *args, **kwargs)
    self.varnames2var[v.name] = v
    return v

  def new_constraint(self, sense, rhs, name):
    c = Constraint(sense, rhs, name=name)
    self.add_constraint(c)
    return c

  def new_objective(self, name):
    o = Objective(name=name)
    self.objectives.append(o)
    return o

  def add_constraint(self, c):
    self.constraints.append(c)


def compute_max(exprs, var_name, Ls, Us, mip):
  """
    Let U' = max(Us)
    expr[i] must lie between Ls[i] and Us[i]
    y >= expr_i
    y <= expr_i + (U' - L_i)* (1 - d_i)
    sum d_i = 1
  """
  U1 = max(Us)
  ds = []
  for i in range(len(exprs)):
    d = mip.new_variable('%s/helper/d%d' % (var_name, i), 0, 1)
    ds.append(d)

  y = mip.new_variable(var_name)

  # y >= expr_i
  # expr_i - y <= 0
  for expr in exprs:
    c = expr.to_constraint('LE', 0)
    c.add_term(y.name, -1)
    mip.add_constraint(c)

  # y <= expr + (U' - L_i)* (1 - d2)
  # expr - y + (U' - L_i)* (1 - d2) >= 0
  # expr - y + (L_i - U')* d2 >= L_i - U'
  for expr, d, l in zip(exprs, ds, Ls):
    c = expr.to_constraint('GE', l - U1)
    c.add_terms([y.name, d.name], [-1, (l - U1)])
    mip.add_constraint(c)

  # sum d_i = 1
  c = Constraint('E', 1)
  c.add_terms([d.name for d in ds], [1] * len(ds))
  mip.add_constraint(c)
  return y


def compute_min(exprs, var_name, Ls, Us, mip):
  """
    Let L' = min(Ls)
    expr[i] must lie between Ls[i] and Us[i]
    y <= expr_i
    y >= expr_i - (U_i - L')* (1 - d_i)
    sum d_i = 1
  """
  L1 = min(Ls)
  ds = []
  for i in range(len(exprs)):
    d = mip.new_variable('%s/helper/d%d' % (var_name, i), 0, 1)
    ds.append(d)

  y = mip.new_variable(var_name)

  # y <= expr_i
  # expr_i - y >= 0
  for expr in exprs:
    c = expr.to_constraint('GE', 0)
    c.add_term(y.name, -1)
    mip.add_constraint(c)

  # y >= expr - (U_i - L')* (1 - d_i)
  # expr - y - (U_i - L')* (1 - d_i) <= 0
  # expr - y + (U_i - L')* d_i <= U_i - L'
  for expr, d, u in zip(exprs, ds, Us):
    c = expr.to_constraint('LE', u - L1)
    c.add_terms([y.name, d.name], [-1, (u - L1)])
    mip.add_constraint(c)

  # sum d_i = 1
  c = Constraint('E', 1)
  c.add_terms([d.name for d in ds], [1] * len(ds))
  mip.add_constraint(c)
  return y


def compute_relu(expr, var_name, L, U, mip):
  return compute_max([expr, Expression(0)], var_name, [L, 0], [U, 0], mip)
