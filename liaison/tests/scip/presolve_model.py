import datetime
import gzip
import pdb
import pickle
import uuid

import numpy as np
import scipy.sparse as sp

import pyscipopt as scip
from absl import app
from liaison.daper.milp.generate_graph import generate_instance
from liaison.daper.milp.primitives import (BinaryVariable, ContinuousVariable,
                                           IntegerVariable, MIPInstance)
from liaison.tests.scip.features import init_scip_params


def add_instance(solver):
  mip = generate_instance('cauction', 50, np.random.RandomState(42))
  mip.add_to_scip_solver(solver)


def get_model():
  m = scip.Model()
  m.hideOutput()
  m.setIntParam('display/verblevel', 0)
  init_scip_params(m, seed=42)
  m.setIntParam('timing/clocktype', 2)
  m.setRealParam('limits/time', 120)
  m.setParam('limits/nodes', 1)
  return m


def test_presolve():
  m = get_model()
  add_instance(m)
  print(len(m.getConss()), len(m.getVars()))
  m.presolve()
  print(len(m.getConss()), len(m.getVars(transformed=True)))
  fname = f'/tmp/model-{uuid.uuid4()}.cip'
  m.writeProblem(fname, True)
  m.freeProb()

  # reinitialize with the transformed model.
  m = get_model()
  m.readProblem(fname)

  mip = MIPInstance()
  for v in m.getVars():
    if v.vtype() == 'BINARY':
      var = BinaryVariable(v.name)
    elif v.vtype() == 'INTEGER':
      var = IntegerVariable(v.name, v.getLbGlobal(), v.getUbGlobal())
    else:
      assert v.vtype() == 'CONTINUOUS'
      var = ContinuousVariable(v.name, v.getLbGlobal(), v.getUbGlobal())
    mip.add_variable(var)

  m.presolve()
  m.optimize()

  for c in m.getConss():
    pdb.set_trace()

  mip.validate()


def main(_):
  test_presolve()


if __name__ == '__main__':
  app.run(main)
