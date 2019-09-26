  def _compute_relu(self, expr, var_name, L, U):
    """
      expr is guaranteed to always be between L and U (inclusive)
      Let U' = max(0, U)
      Returns variable y = abs(expr) using the following constraints
      y >= 0
      y >= expr
      y <= U'* (1 - d1)
      y <= expr + (U' - L) * (1 - d2)
      d1 + d2 = 1
    """
    U1 = max(U, 0)
    d1 = Variable('%s/helper/d1' % var_name, 0, 1, self._variable_constraints)
    d2 = Variable('%s/helper/d2' % var_name, 0, 1, self._variable_constraints)
    y = Variable(var_name, 0, None, self._variable_constraints)
    self._varnames2var['%s/helper/d1' % var_name] = d1
    self._varnames2var['%s/helper/d2' % var_name] = d2
    self._varnames2var[var_name] = y

    # y >= expr
    # expr - y <= 0
    c = expr.to_constraint('LE', 0)
    c.add_term(y.name, -1)
    self._misc_constraints.append(c)

    # y <= U'* (1 - d1)
    # y + U'* d1 <= U'
    c = Constraint('LE', U1)
    c.add_terms([y.name, d1], [1, U1])
    self._misc_constraints.append(c)

    # y <= expr + (U' - L)* (1 - d2)
    # expr - y + (U' - L)* (1 - d2) >= 0
    # expr - y + (L - U')* d2 >= L - U'
    c = expr.to_constraint('GE', L - U1)
    c.add_terms([y.name, d2.name], [-1, (L - U1)])
    self._misc_constraints.append(c)

    # d1 + d2 = 1
    c = Constraint('E', 1)
    c.add_terms([d1.name, d2.name], [1, 1])
    self._misc_constraints.append(c)
    return y
