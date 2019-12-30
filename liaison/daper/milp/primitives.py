import copy
import os
from abc import ABC
from typing import Any, Dict, Text, Tuple, Union

from pyscipopt import Model, multidict, quicksum


class Variable(ABC):

  def __init__(self, var_name, lower_bound=None, upper_bound=None):
    self.name = var_name
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound

  def integral_relax(self):
    """relax the integrality constraint for lp."""
    return ContinuousVariable(self.name, self.lower_bound, self.upper_bound)

  def validate(self, val):
    if self.lower_bound is not None:
      assert val >= self.lower_bound - 1e-5, (val, self.lower_bound)

    if self.upper_bound is not None:
      assert val <= self.upper_bound + 1e-5, (val, self.upper_bound)

    return True

  def is_integer(self):
    return isinstance(self, IntegerVariable)


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

  def __str__(self):
    ret = ''
    if self.constant > 0:
      ret += f'{self.constant} + '

    for i, (v, c) in enumerate(zip(self.var_names, self.coeffs)):
      ret += f'{c} * {v}'
      if i != len(self) - 1:
        ret += ' + '
    return ret


class Constraint:

  def __init__(self, sense, rhs, name=None):
    """
      senses: "LE", "GE"
    """
    assert sense in ['LE', 'GE']
    self.name = name
    self.sense = sense
    self.rhs = float(rhs)
    self.expr = Expression()

  def add_term(self, var_name, coeff):
    self.expr.add_term(var_name, coeff)

  def add_terms(self, var_names, coeffs):
    self.expr.add_terms(var_names, coeffs)

  def __len__(self):
    return len(self.expr)

  def cast_sense_to_le(self):
    rhs = self.rhs
    expr = copy.deepcopy(self.expr)
    if self.sense == 'GE':
      expr = expr.negate()
      rhs *= -1
    c = Constraint('LE', rhs, self.name)
    c.expr = expr
    c.validate()
    return c

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
        assert expr.constant <= self.rhs + 1e-4, (expr.constant, self.rhs,
                                                  os.getpid())
      else:
        assert expr.constant >= self.rhs - 1e-4, (expr.constant, self.rhs,
                                                  os.getpid())
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
      c.expr = copy.deepcopy(expr)
      c.validate()
      return c

  def validate(self):
    assert self.expr.constant == 0
    return True

  def validate_assignment(self, ass: Dict[str, float]):
    c = self.relax(ass)
    assert c is None
    return True

  def __str__(self):
    ret = ''
    if self.name:
      ret += f'Name of the constraint: {self.name}\n'
    else:
      ret += 'Unnamed constraint:\n'

    ret += str(self.expr)
    ret += f' {self.sense} {self.rhs}'
    return ret

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

  def __init__(self, name=None, constant=0):
    self.name = name
    self.expr = Expression(constant=constant)

  def add_term(self, var_name, coeff):
    self.expr.add_term(var_name, coeff)

  def add_terms(self, var_names, coeffs):
    self.expr.add_terms(var_names, coeffs)

  def __len__(self):
    return len(self.expr)

  def __str__(self):
    ret = f'Objective Name: {self.name}\n'
    ret += str(self.expr)
    return ret

  def relax(self, fixed_vars_to_values):
    """
      returns objective after removing the fixed variables.
      returns None if all variables get eliminated and objective is trivial.
    """
    o = Objective()
    expr = self.expr.reduce(fixed_vars_to_values)
    o.expr = expr

    if len(o.expr) == 0:
      return None
    return o

  def add_to_cplex_solver(self, solver):
    solver.objective.set_sense(solver.objective.sense.minimize)
    solver.objective.set_linear(zip(self.expr.var_names, self.expr.coeffs))
    solver.objective.set_offset(self.expr.constant)

  def add_to_scip_solver(self, solver, varname2var):
    solver.setObjective(
        quicksum(
            (varname2var[var] * coeff
             for var, coeff in zip(self.expr.var_names, self.expr.coeffs))),
        "minimize")
    solver.addObjoffset(self.expr.constant)


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

  def old_fix(self,
              fixed_ass: Dict[str, float],
              relax_integral_constraints=False):
    """
      Args:
        fixed_vars_to_values: variables to fix and their values to fix to.
        integral_relax: If true return an lp with all integral constraints on
                        all variables relaxed.
      Returns:
        Leaves the current mipinstance unchanged (immutable call).
        Returns a new mipinstance with fixes and relaxations made.
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
        if integral_relax:
          m.add_variable(var.integral_relax())
        else:
          m.add_variable(copy.deepcopy(var))

    m.obj = self.obj.relax(fixed_vars_to_values)
    assert m.obj is not None
    m.validate()
    return m

  def fix(self, fixed_ass: Dict[str, float], relax_integral_constraints=False):
    """
      Args:
        fixed_vars_to_values: variables to fix and their values to fix to.
        integral_relax: If true return an lp with all integral constraints on
                        all variables relaxed.
      Returns:
        Leaves the current mipinstance unchanged (immutable call).
        Returns a new mipinstance with fixes and relaxations made.
    """
    for name, val in fixed_ass.items():
      # variables are defined.
      assert name in self.varname2var
      self.varname2var[name].validate(val)

    if self.name:
      m = MIPInstance(self.name + '-relaxed')
    else:
      m = MIPInstance()

    m.constraints = copy.deepcopy(self.constraints)
    # add fixed assignment constraints
    for var, val in fixed_ass.items():
      c1 = m.new_constraint('LE',
                            val + 1e-3,
                            name=f'fix-var-{var}-to-LE-{val}')
      c1.add_term(var, 1)
      c2 = m.new_constraint('GE',
                            val - 1e-3,
                            name=f'fix-var-{var}-to-GE-{val}')
      c2.add_term(var, 1)

    for vname, var in self.varname2var.items():
      if relax_integral_constraints:
        m.add_variable(var.integral_relax())
      else:
        m.add_variable(copy.deepcopy(var))

    m.obj = copy.deepcopy(self.obj)
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

  def validate_sol(self, solution: Dict[str, float]):

    for k in self.varname2var:
      # assert solution is the full solution.
      assert k in solution

    for c in self.constraints:
      c.validate_assignment(solution)

    for k, v in solution.items():
      assert self.varname2var[k].validate(v)
    return True
