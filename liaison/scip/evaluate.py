# Evaluates the agent inside SCIP
import copy
import math
import pickle
import sys
import time
from math import fabs
from multiprocessing.pool import ThreadPool
from threading import Thread

import numpy as np

import tree as nest
from liaison.daper.milp.heuristics.heuristic_fn import run as heuristic_run
from liaison.daper.milp.primitives import IntegerVariable
from liaison.env import StepType
from liaison.env.batch import ParallelBatchedEnv, SerialBatchedEnv
from liaison.env.utils.rins import get_sample
from liaison.scip.scip_integration import (EvalHeur, init_scip_params,
                                           run_branch_and_bound_scip)
from liaison.utils import ConfigDict
from pyscipopt import Model

EPS = 1e-2


def get_model(seed, gap=0.0, nodes=None, heur_frequency=-1):
  model = Model()
  # model.hideOutput()
  model.setIntParam("display/verblevel", 4)
  init_scip_params(model, seed=seed, presolving=True, heur_frequency=heur_frequency)
  model.setRealParam('limits/gap', gap)
  if nodes is not None:
    model.setParam('limits/nodes', nodes)
  return model


class Evaluator:
  """
    Evaluator is responsible for the following.

    (1) Create a shell and batched environments.
    (2) Pushes collected evaluations to irs.
  """

  def __init__(
      self,
      shell_class,
      shell_config,
      agent_class,
      agent_config,
      env_class,
      env_config,
      seed,
      dataset,
      dataset_type,
      graph_start_idx,
      gap,
      max_nodes,
      heur_frequency,
      batch_size=1,  # num_envs
      use_parallel_envs=False,
      use_threaded_envs=False,
      **config):
    self.config = ConfigDict(config)
    self.batch_size = batch_size
    self.rng = np.random.RandomState(seed)
    self.env_config = env_config
    # get the mip instance
    milp = get_sample(dataset, dataset_type, graph_start_idx)
    model = get_model(seed, gap, max_nodes, heur_frequency)
    milp.mip.add_to_scip_solver(model)
    # presolve isnce the agent uses pre-solved model for its input.
    model.presolve()
    self.k = env_config.k
    self._model = model
    # init shell and environments
    # init environment
    env_configs = []
    for i in range(batch_size):
      env_config_copy = ConfigDict(**env_config)
      env_config_copy.update({
          'graph_start_idx': graph_start_idx,
          'dataset': dataset,
          'dataset_type': dataset_type,
          'n_local_moves': int(1e10),  # infinity
      })
      env_configs.append(env_config_copy)

    milp = get_sample(dataset, dataset_type, graph_start_idx)
    self.mip = milp.mip
    self._optimal_lp_sol = milp.optimal_lp_sol
    if use_parallel_envs:
      self._env = ParallelBatchedEnv(batch_size,
                                     env_class,
                                     env_configs,
                                     seed,
                                     use_threads=use_threaded_envs)
    else:
      self._env = SerialBatchedEnv(batch_size, env_class, env_configs, seed)

    # disable sync
    shell_config['sync_period'] = None
    self._shell = shell_class(
        action_spec=self._env.action_spec(),
        obs_spec=self._env.observation_spec(),
        agent_class=agent_class,
        agent_config=agent_config,
        batch_size=batch_size,
        seed=seed,
        verbose=False,
        **shell_config,
    )

  def run(self, without_scip=False, without_agent=False, heuristic=None):
    if without_scip:
      return self.run_without_scip()
    elif heuristic:
      if heuristic == 'rins':
        self._heuristic = self.rins_fn
      elif heuristic == 'rens':
        self._heuristic = self.rens_fn
      else:
        raise Exception(f'{heuristic} not found!')
      heuristic = EvalHeur(self._heuristic_callback)
    elif without_agent:
      heuristic = EvalHeur(lambda *_, **__: (None, None))
    else:
      heuristic = EvalHeur(self._primal_heuristic_callback)

    results = run_branch_and_bound_scip(self._model, heuristic)
    heuristic.done()
    log_vals = heuristic._obj_vals
    with open(f'{self.config.results_dir}/out.pkl', 'wb') as f:
      pickle.dump(log_vals, f)

  def rins_fn(self, i):
    sol = self._env.func_call_ith_env('get_curr_soln', i)
    var_names = self._env.func_call_ith_env('get_varnames', i)
    errs = []
    for var, val in self._optimal_lp_sol.items():
      if isinstance(self.mip.varname2var[var], IntegerVariable) and var in var_names:
        err = math.fabs(val - sol[var])
        errs.append((err, var))
    errs = sorted(errs, reverse=True)
    var_names = [var for err, var in errs[:self.k]]
    for err, var_name in errs:
      if err == errs[self.k - 1][0]:
        if var_name not in var_names:
          var_names.append(var_name)
    act = []
    for vname in self.rng.choice(var_names, size=self.k, replace=False):
      act.append(self._env.func_call_ith_env('varname2idx', i, vname))
    return act

  def rens_fn(self, i):
    assert self.env_config.use_rens_submip_bounds
    sol = self._env.func_call_ith_env('get_curr_soln', i)
    var_names = self._env.func_call_ith_env('get_varnames', i)
    errs = []
    for var, val in self._optimal_lp_sol.items():
      if isinstance(self.mip.varname2var[var], IntegerVariable) and var in var_names:
        # if lp relaxation is not integral -- consider it to unfix  .
        if math.fabs(math.modf(val)[0]) <= 1e-3:
          err = math.fabs(val - sol[var])
          errs.append((err, var))
    errs = sorted(errs, reverse=True)
    var_names = [var for err, var in errs[:self.k]]
    for err, var_name in errs:
      if err == errs[self.k - 1][0]:
        if var_name not in var_names:
          var_names.append(var_name)
    act = []
    for vname in self.rng.choice(var_names, size=self.k, replace=False):
      act.append(self._env.func_call_ith_env('varname2idx', i, vname))
    return act

  def _heuristic_callback(self, model, sol, obj_val, step):
    ts = self._env.func_call_with_common_args('reset_solution', sol, obj_val)

    start_obj = np.min(ts.observation['curr_episode_log_values']['curr_obj'])
    assert math.fabs(obj_val - start_obj) <= EPS, (obj_val, start_obj, obj_val - start_obj)
    start_qual = np.min(ts.observation['curr_episode_log_values']['best_quality'])
    n_local_moves_start = ts.observation['n_local_moves']
    stats = dict(mip_work=0)
    while np.all(
        ts.observation['n_local_moves'] - n_local_moves_start <= self.config.n_local_moves):
      action = np.asarray([self._heuristic(i) for i in range(self.batch_size)], np.int32)
      for act in np.transpose(action):
        ts = self._env.step(act)
        stats['mip_work'] += ts.observation['curr_episode_log_values']['mip_work']

    # get the best improved solution among all the environments
    sol = self._env.func_call_ith_env(
        'get_curr_soln', np.argmin(ts.observation['curr_episode_log_values']['best_quality']))
    assert isinstance(sol, dict)

    final_qual = np.min(ts.observation['curr_episode_log_values']['best_quality'])
    final_obj = np.min(ts.observation['curr_episode_log_values']['curr_obj'])

    # solution improved.
    if final_obj < start_obj:
      print(f'Start objective: {start_obj} Final Objective: {final_obj}')
      print(f'Start Quality: {start_qual} Final Quality: {final_qual}')
      return sol, stats
    else:
      return None, stats

  def run_without_scip(self):
    # run as a local-move solver independent of SCIP.
    ts = self._env.reset()
    start_obj = np.min(ts.observation['curr_episode_log_values']['curr_obj'])
    start_qual = np.min(ts.observation['curr_episode_log_values']['best_quality'])
    stats = dict(
        obj=[np.min(ts.observation['curr_episode_log_values']['curr_obj'])],
        quality=[np.min(ts.observation['curr_episode_log_values']['best_quality'])],
    )
    while np.all(ts.observation['n_local_moves'] <= self.config.n_local_moves):
      step_output = self._shell.step(step_type=ts.step_type,
                                     reward=ts.reward,
                                     observation=ts.observation)
      ts = self._env.step(step_output.action)
      stats['obj'].append(np.min(ts.observation['curr_episode_log_values']['curr_obj']))
      stats['quality'].append(np.min(ts.observation['curr_episode_log_values']['best_quality']))

    for k, l in stats.items():
      print(f'{k} improved from {l[0]} to {l[-1]}')
    with open(f'{self.config.results_dir}/out.pkl', 'wb') as f:
      pickle.dump(stats, f)

  def _primal_heuristic_callback(self, _, sol, obj_val, step):
    ts = self._env.func_call_with_common_args('reset_solution', sol, obj_val)

    start_obj = np.min(ts.observation['curr_episode_log_values']['curr_obj'])
    assert math.fabs(obj_val - start_obj) <= EPS, (obj_val, start_obj, obj_val - start_obj)
    start_qual = np.min(ts.observation['curr_episode_log_values']['best_quality'])
    n_local_moves_start = ts.observation['n_local_moves']
    stats = dict(mip_work=0)
    while np.all(
        ts.observation['n_local_moves'] - n_local_moves_start <= self.config.n_local_moves):
      step_output = self._shell.step(step_type=ts.step_type,
                                     reward=ts.reward,
                                     observation=ts.observation)
      ts = self._env.step(step_output.action)
      stats['mip_work'] += ts.observation['curr_episode_log_values']['mip_work']

    # get the best improved solution among all the environments
    sol = self._env.func_call_ith_env(
        'get_curr_soln', np.argmin(ts.observation['curr_episode_log_values']['best_quality']))
    assert isinstance(sol, dict)

    final_qual = np.min(ts.observation['curr_episode_log_values']['best_quality'])
    final_obj = np.min(ts.observation['curr_episode_log_values']['curr_obj'])

    # solution improved.
    if final_obj < start_obj:
      print(f'Start objective: {start_obj} Final Objective: {final_obj}')
      print(f'Start Quality: {start_qual} Final Quality: {final_qual}')
      return sol, stats
    else:
      return None, stats
