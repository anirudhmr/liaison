import functools
import os
import pickle

import numpy as np
from liaison.daper.dataset_constants import DATASET_PATH
from liaison.daper.milp.primitives import relax_integral_constraints
from liaison.utils import ConfigDict
from pyscipopt import Model


def pad_first_dim(features: np.ndarray, pad_to_len):
  """pads first dim to `pad_to_len`."""
  if pad_to_len < 0:
    return features
  assert features.shape[0] <= pad_to_len, (features.shape, pad_to_len)
  return np.pad(features, [(0, pad_to_len - features.shape[0])] + [(0, 0)] *
                (features.ndim - 1))


def pad_last_dim(features: np.ndarray, pad_to_len):
  """pads last dim to `pad_to_len`."""
  if pad_to_len < 0:
    return features

  assert features.shape[-1] <= pad_to_len, (features.shape, pad_to_len)
  return np.pad(features, [(0, 0)] * (features.ndim - 1) +
                [(0, pad_to_len - features.shape[-1])])


@functools.lru_cache(maxsize=100)
def get_sample(dataset, dataset_type, graph_idx):
  dataset_path = DATASET_PATH[dataset]

  with open(os.path.join(dataset_path, dataset_type, f'{graph_idx}.pkl'),
            'rb') as f:
    milp = pickle.load(f)

  solver = Model()
  solver.hideOutput()
  relax_integral_constraints(milp.mip).add_to_scip_solver(solver)
  solver.optimize()
  assert solver.getStatus() == 'optimal', solver.getStatus()
  ass = {var.name: solver.getVal(var) for var in solver.getVars()}
  milp = ConfigDict(milp)
  milp.optimal_lp_sol = ass
  return milp
