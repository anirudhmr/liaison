import functools
import os
import pickle
import time

import numpy as np
from liaison.daper.dataset_constants import DATASET_INFO_PATH, DATASET_PATH
from liaison.daper.milp.primitives import relax_integral_constraints
from liaison.distributed import ParameterClient
from liaison.utils import ConfigDict
from pyscipopt import Model


def pad_first_dim(features: np.ndarray, pad_to_len):
  """pads first dim to `pad_to_len`."""
  if pad_to_len < 0:
    return features
  assert features.shape[0] <= pad_to_len, (features.shape, pad_to_len)
  return np.pad(features, [(0, pad_to_len - features.shape[0])] + [(0, 0)] * (features.ndim - 1),
                mode='constant')


def pad_last_dim(features: np.ndarray, pad_to_len):
  """pads last dim to `pad_to_len`."""
  if pad_to_len < 0:
    return features

  assert features.shape[-1] <= pad_to_len, (features.shape, pad_to_len)
  return np.pad(features, [(0, 0)] * (features.ndim - 1) + [(0, pad_to_len - features.shape[-1])],
                mode='constant')


@functools.lru_cache(maxsize=100)
def get_sample(dataset, dataset_type, graph_idx):
  dataset_path = DATASET_PATH[dataset]

  with open(os.path.join(dataset_path, dataset_type, f'{graph_idx}.pkl'), 'rb') as f:
    milp = pickle.load(f)

  if milp.get('optimal_lp_sol', None) is None:
    if dataset in DATASET_INFO_PATH:
      with open(
          os.path.join(DATASET_INFO_PATH[dataset], 'aux_info', dataset_type, f'{graph_idx}.pkl'),
          'rb') as f:
        pkl = pickle.load(f)
        milp.optimal_lp_sol = pkl['optimal_lp_sol']
    else:
      solver = Model()
      solver.hideOutput()
      relax_integral_constraints(milp.mip).add_to_scip_solver(solver)
      solver.optimize()
      assert solver.getStatus() == 'optimal', solver.getStatus()
      ass = {var.name: solver.getVal(var) for var in solver.getVars()}
      milp = ConfigDict(milp)
      milp.optimal_lp_sol = ass
  return milp


@functools.lru_cache(maxsize=1000)
def get_hamming_dists(dataset, dataset_type, graph_idx):
  with open(
      os.path.join(DATASET_INFO_PATH[dataset], 'hamming_distance', dataset_type,
                   f'{graph_idx}.pkl'), 'rb') as f:
    return pickle.load(f)


@functools.lru_cache(maxsize=1000)
def load_pickled_features(dataset, dataset_type, graph_idx):
  with open(os.path.join(DATASET_INFO_PATH[dataset], 'aux_info', dataset_type, f'{graph_idx}.pkl'),
            'rb') as f:
    return pickle.load(f)['mip_features']


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


def linear_interpolate_inc(step, start_step, dec_steps, s_v, max_v):
  val = s_v
  val += (step - start_step) * (max_v - s_v) / dec_steps
  val = min(val, max_v)
  return val


def linear_interpolate_dec(step, start_step, dec_steps, s_v, min_v):
  val = s_v
  val += (step - start_step) * (min_v - s_v) / dec_steps
  val = max(val, min_v)
  return val
