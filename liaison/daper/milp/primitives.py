class ContinuousVariable:

  def __init__(self, var_name, lower_bound=None, upper_bound=None):
    self.name = var_name
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound

  def add_to_cplex_solver(self, solver):
    solver.variables.add(names=[self.name])
    solver.variables.set_types(self.name, solver.variables.type.continuous)
    if self.lower_bound is not None:
      solver.variables.set_lower_bounds(self.name, self.lower_bound)
    if self.upper_bound is not None:
      solver.variables.set_upper_bounds(self.name, self.upper_bound)


class IntegerVariable:

  def __init__(self, var_name, lower_bound=None, upper_bound=None):
    self.name = var_name
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound

  def add_to_cplex_solver(self, solver):
    solver.variables.add(names=[self.name])
    solver.variables.set_types(self.name, solver.variables.type.integer)
    if self.lower_bound is not None:
      solver.variables.set_lower_bounds(self.name, self.lower_bound)
    if self.upper_bound is not None:
      solver.variables.set_upper_bounds(self.name, self.upper_bound)


class BinaryVariable(IntegerVariable):

  def __init__(self, var_name):
    super(BinaryVariable, self).__init__(var_name,
                                         lower_bound=0,
                                         upper_bound=1)

  def add_to_cplex_solver(self, solver):
    solver.variables.add(names=[self.name])
    solver.variables.set_types(self.name, solver.variables.type.binary)


class Expression:

  def __init__(self, constant=0):
    self.var_names = []
    self.coeffs = []
    self.constant = constant

  def add_term(self, var_name, coeff):
    assert isinstance(var_name, str)
    self.var_names.append(var_name)
    self.coeffs.append(coeff)

  def add_terms(self, var_names, coeffs):
    assert isinstance(var_names[0], str)
    self.var_names.extend(var_names)
    self.coeffs.extend(coeffs)


class Constraint:

  def __init__(self, sense, rhs, name=None):
    """
      senses: "E", "LE", "GE"
    """
    self.sense = sense
    self.rhs = float(rhs)
    self.expr = Expression()

  def add_term(self, var_name, coeff):
    self.expr.add_term(var_name, coeff)

  def add_terms(self, var_names, coeffs):
    self.expr.add_terms(var_names, coeffs)

  def add_to_cplex_solver(self, solver):
    sense = self.sense
    if sense == 'LE':
      sense = 'L'  # different terminology
    elif sense == 'GE':
      sense = 'G'

    import cplex
    solver.linear_constraints.add(lin_expr=[
        cplex.SparsePair(ind=self.expr.var_names, val=self.expr.coeffs)
    ],
                                  senses=[sense],
                                  rhs=[self.rhs])


class Objective:
  """Always minimize the objective."""

  def __init__(self, name=None):
    self.expr = Expression()

  def add_term(self, var_name, coeff):
    self.expr.add_term(var_name, coeff)

  def add_terms(self, var_names, coeffs):
    self.expr.add_terms(var_names, coeffs)

  def add_to_cplex_solver(self, solver):
    solver.objective.set_sense(solver.objective.sense.minimize)
    solver.objective.set_linear(zip(self.expr.var_names, self.expr.coeffs))


class MIPInstance:

  def __init__(self, name=None):
    self.varname2var = dict()
    self.constraints = []
    self.obj = Objective()

  def add_variable(self, var):
    self.varname2var[var.name] = var

  def new_constraint(self, *args, **kwargs):
    c = Constraint(*args, **kwargs)
    self.constraints.append(c)
    return c

  def get_objective(self):
    return self.obj

  def validate(self):
    """
      Check if variables referenced in the constraints and objectives are
      actually created in vars.
    """
    all_var_names = []
    for c in self.constraints:
      all_var_names += c.expr.var_names
    all_var_names += self.obj.expr.var_names

    assert set(all_var_names) == set(list(self.varname2var.keys()))

  def add_to_cplex_solver(self, solver):
    """
      Args:
        solver: cplex.Cplex() object.
    """
    self.validate()

    # add variables
    for v in self.varname2var.values():
      v.add_to_cplex_solver(solver)

    # add constraints
    for c in self.constraints:
      c.add_to_cplex_solver(solver)

    self.obj.add_to_cplex_solver(solver)
