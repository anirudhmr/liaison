from pyscipopt import Model, multidict, quicksum


class Variable:

  def __init__(self, var_name, lower_bound=None, upper_bound=None):
    self.name = var_name
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound

  def integral_relax(self):
    """relax the integrality constraint for lp."""
    return ContinuousVariable(self.name, self.lower_bound, self.upper_bound)

  def validate(self, val):
    if self.lower_bound is not None:
      assert val >= self.lower_bound

    if self.upper_bound is not None:
      assert val <= self.upper_bound

    return True


class ContinuousVariable(Variable):

  def add_to_cplex_solver(self, solver):
    solver.variables.add(names=[self.name])
    solver.variables.set_types(self.name, solver.variables.type.continuous)
    if self.lower_bound is not None:
      solver.variables.set_lower_bounds(self.name, self.lower_bound)
    if self.upper_bound is not None:
      solver.variables.set_upper_bounds(self.name, self.upper_bound)

  def add_to_scip_solver(self, solver):
    return solver.addVar(lb=self.lower_bound,
                         ub=self.upper_bound,
                         vtype="C",
                         name=self.name)


class IntegerVariable(Variable):

  def validate(self, val):
    assert isinstance(val, int)
    super().validate(val)

  def add_to_cplex_solver(self, solver):
    solver.variables.add(names=[self.name])
    solver.variables.set_types(self.name, solver.variables.type.integer)
    if self.lower_bound is not None:
      solver.variables.set_lower_bounds(self.name, self.lower_bound)
    if self.upper_bound is not None:
      solver.variables.set_upper_bounds(self.name, self.upper_bound)

  def add_to_scip_solver(self, solver):
    return solver.addVar(lb=self.lower_bound,
                         ub=self.upper_bound,
                         vtype="I",
                         name=self.name)


class BinaryVariable(IntegerVariable):

  def __init__(self, var_name):
    super(BinaryVariable, self).__init__(var_name,
                                         lower_bound=0,
                                         upper_bound=1)

  def validate(self, val):
    assert isinstance(val, int)
    super().validate(val)

  def add_to_cplex_solver(self, solver):
    solver.variables.add(names=[self.name])
    solver.variables.set_types(self.name, solver.variables.type.binary)

  def add_to_scip_solver(self, solver):
    return solver.addVar(lb=self.lower_bound,
                         ub=self.upper_bound,
                         vtype="B",
                         name=self.name)


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

  def __len__(self):
    assert len(self.var_names) == len(self.coeffs)
    return len(self.var_names)

  @property
  def is_constant(self):
    return len(self.var_names) == 0

  def negate(self):
    e = Expression(-self.constant)
    e.add_terms(self.var_names, [-1 * c for c in self.coeffs])
    return e

  def reduce(self, fixed_vars_to_values):
    """
      Returns new expression with fixed_vars eliminated by assigning them
      the given fixed values.
      Args:
        fixed_vars_to_values: Dict from var_names to their values.
    """
    e = Expression()
    reduced_val = self.constant
    for var, coeff in zip(self.var_names, self.coeffs):
      if var in fixed_vars_to_values:
        reduced_val += (fixed_vars_to_values[var] * coeff)
      else:
        e.add_term(var, coeff)
    e.constant = reduced_val
    return e


class Constraint:

  def __init__(self, sense, rhs, name=None):
    """
      senses: "LE", "GE"
    """
    assert sense in ['LE', 'GE']
    self.sense = sense
    self.rhs = float(rhs)
    self.expr = Expression()

  def add_term(self, var_name, coeff):
    self.expr.add_term(var_name, coeff)

  def add_terms(self, var_names, coeffs):
    self.expr.add_terms(var_names, coeffs)

  def relax(self, fixed_vars_to_values):
    """
      returns constraints after removing the fixed variables.
      returns None if all variables get eliminated and constraint
                   becomes trivially satisfied.
      raises AssertError if constraint becomes unsatisfiable.
    """
    expr = self.expr.reduce(fixed_vars_to_values)
    if expr.is_constant:
      if self.sense == 'LE':
        assert expr.constant <= self.rhs
      else:
        assert expr.constant >= self.rhs
      return None
    else:
      # convert to 'LE' format
      if self.sense == 'GE':
        expr = expr.negate()
        rhs = -expr.constant - self.rhs
      else:
        rhs = self.rhs - expr.constant

      # expr constant has been absorbed into rhs
      expr.constant = 0

      c = Constraint('LE', rhs)
      c.expr = expr
      c.validate()
      return c

  def validate(self):
    assert self.expr.constant == 0

  def add_to_cplex_solver(self, solver):
    self.validate()
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

  def add_to_scip_solver(self, solver, varname2var):
    if self.sense == 'LE':
      solver.addCons(
          quicksum((varname2var[var] * coeff for var, coeff in zip(
              self.expr.var_names, self.expr.coeffs))) <= self.rhs)
    elif self.sense == 'GE':
      solver.addCons(
          quicksum((varname2var[var] * coeff for var, coeff in zip(
              self.expr.var_names, self.expr.coeffs))) >= self.rhs)


class Objective:
  """Always minimize the objective."""

  def __init__(self, name=None):
    self.expr = Expression()

  def add_term(self, var_name, coeff):
    self.expr.add_term(var_name, coeff)

  def add_terms(self, var_names, coeffs):
    self.expr.add_terms(var_names, coeffs)

  def relax(self, fixed_vars_to_values):
    """
      returns objective after removing the fixed variables.
      returns None if all variables get eliminated and objective is trivial.
    """
    o = Objective()
    for var, coeff in zip(self.expr.var_names, self.expr.coeffs):
      if var in fixed_vars_to_values:
        # ignore the fixed vars
        pass
      else:
        o.add_term(var, coeff)

    if len(o.expr) == 0:
      return None
    return o

  def add_to_cplex_solver(self, solver):
    solver.objective.set_sense(solver.objective.sense.minimize)
    solver.objective.set_linear(zip(self.expr.var_names, self.expr.coeffs))

  def add_to_scip_solver(self, solver, varname2var):
    solver.setObjective(
        quicksum(
            (varname2var[var] * coeff
             for var, coeff in zip(self.expr.var_names, self.expr.coeffs))),
        "minimize")


class MIPInstance:

  def __init__(self, name=None):
    self.varname2var = dict()
    self.constraints = []
    self.name = name
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

  def relax(self, fixed_vars_to_values):
    """returns lp version of mip without integer variables present
       in fixed_vars_to_values.
    """
    for name, val in fixed_vars_to_values.items():
      # assert variables are defined.
      assert name in self.varname2var
      self.varname2var[name].validate(val)

    if self.name:
      m = MIPInstance(self.name + '-relaxed')
    else:
      m = MIPInstance()

    for c in self.constraints:
      new_constraint = c.relax(fixed_vars_to_values)
      if new_constraint:
        m.constraints.append(new_constraint)

    for vname, var in self.varname2var.items():
      if vname in fixed_vars_to_values:
        # ignore this variable since it is fixed
        # and eliminated in the sub-MIP
        pass
      else:
        m.add_variable(var.integral_relax())

    m.obj = self.obj.relax(fixed_vars_to_values)
    assert m.obj is not None
    m.validate()
    return m

  def add_to_scip_solver(self, solver):
    self.validate()

    # add variables
    varname2scipvar = dict()
    for vname, v in self.varname2var.items():
      varname2scipvar[vname] = v.add_to_scip_solver(solver)

    # add constraints
    for c in self.constraints:
      c.add_to_scip_solver(solver, varname2scipvar)

    self.obj.add_to_scip_solver(solver, varname2scipvar)

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
