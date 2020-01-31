import functools
import os
import pickle
import time

import numpy as np

from liaison.daper.dataset_constants import DATASET_PATH
from liaison.daper.milp.primitives import relax_integral_constraints
from liaison.distributed import ParameterClient
from liaison.utils import ConfigDict
from pyscipopt import Model


def pad_first_dim(features: np.ndarray, pad_to_len):
  """pads first dim to `pad_to_len`."""
  if pad_to_len < 0:
    return features
  assert features.shape[0] <= pad_to_len, (features.shape, pad_to_len)
  return np.pad(features, [(0, pad_to_len - features.shape[0])] + [(0, 0)] *
                (features.ndim - 1),
                mode='constant')


def pad_last_dim(features: np.ndarray, pad_to_len):
  """pads last dim to `pad_to_len`."""
  if pad_to_len < 0:
    return features

  assert features.shape[-1] <= pad_to_len, (features.shape, pad_to_len)
  return np.pad(features, [(0, 0)] * (features.ndim - 1) +
                [(0, pad_to_len - features.shape[-1])],
                mode='constant')


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


class GlobalStepFetcher:
  # fetches global step value from the parameter server.
  # caches to avoid overloading the remote server.

  def __init__(self, min_request_spacing=2):
    self._ps_client = ParameterClient(host=os.environ['SYMPH_PS_SERVING_HOST'],
                                      port=os.environ['SYMPH_PS_SERVING_PORT'],
                                      agent_scope=None,
                                      timeout=2,
                                      not_ready_sleep=2)
    self._min_request_spacing = min_request_spacing
    self._prev_time = time.time()
    self._prev_response = 0

  def get(self):
    if time.time() - self._prev_time >= self._min_request_spacing:
      info = self._ps_client.fetch_info_no_retry()
      if info:
        self._prev_response = info['iteration']
      self._prev_time = time.time()
    return self._prev_response


def np_slack_down(a):
  # a is np array
  return a - np.floor(a)


def np_slack_up(a):
  # a is np array
  return np.ceil(a) - a
