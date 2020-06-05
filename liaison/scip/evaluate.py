# Evaluates the agent inside SCIP
import functools
import math
import pickle
from math import fabs
from multiprocessing.pool import ThreadPool

import numpy as np
import tree as nest
from liaison.daper.milp.heuristics.heuristic_fn import scip_solve
from liaison.daper.milp.primitives import IntegerVariable
from liaison.env.batch import ParallelBatchedEnv, SerialBatchedEnv
from liaison.env.utils.rins import get_sample
from liaison.scip.scip_integration import (EvalHeur, init_scip_params,
                                           run_branch_and_bound_scip)
from liaison.utils import ConfigDict
from pyscipopt import Model

EPS = 1e-2
N_EARLY_STOP = 3


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
      batch_size=1,  # num_envs
      use_parallel_envs=False,
      use_threaded_envs=False,
      create_shell=True,
      **config):
    self.config = ConfigDict(seed=seed, **config)
    self.batch_size = batch_size
    self.rng = np.random.RandomState(seed)
    self.env_config = env_config
    # get the mip instance
    self.k = env_config.k
    # init shell and environments
    # init environment
    env_configs = []
    for _ in range(batch_size):
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

    if create_shell:
      # disable sync
      shell_config['sync_period'] = None
      assert shell_config['restore_from']
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

  def _early_stop(self, objs_d):
    objs = sorted(list(objs_d.keys()))
    if len(objs) <= N_EARLY_STOP:
      return False

    for idx in range(-2, -N_EARLY_STOP - 1, -1):
      if fabs(objs_d[objs[-1]]['obj'] - objs_d[objs[idx]]['obj']) >= EPS:
        return False
    return True

  def run(self, standalone=False, without_agent=False, heuristic=None):
    config = self.config
    self._heuristic = None
    if heuristic == 'rins':
      self._heuristic = self.rins_fn
    elif heuristic == 'least_integral':
      self._heuristic = functools.partial(self.integral, least_integral=True)
    elif heuristic == 'most_integral':
      self._heuristic = functools.partial(self.integral, least_integral=False)
    elif heuristic == 'random':
      self._heuristic = self.random_fn
    elif heuristic != None:
      raise Exception(f'{heuristic} not found!')

    # if standalone run and exit function.
    if standalone:
      return self.run_standalone(self._heuristic_fn if self._heuristic else None)

    if heuristic:
      heuristic = EvalHeur(
          functools.partial(self._scip_callback,
                            heuristic_fn=self._heuristic_fn if self._heuristic else None))
    elif without_agent:
      heuristic = EvalHeur(lambda *_, **__: (None, None))
    else:
      heuristic = EvalHeur(self._scip_callback)

    model = get_model(config.seed, config.gap, config.max_nodes, config.heur_frequency)
    self.mip.add_to_scip_solver(model)
    # presolve isnce the agent uses pre-solved model for its input.
    model.presolve()
    run_branch_and_bound_scip(model, heuristic)
    heuristic.done()
    log_vals = heuristic._obj_vals
    with open(f'{self.config.results_dir}/out.pkl', 'wb') as f:
      pickle.dump(log_vals, f)

  def rins_fn(self, sol, var_names):
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
    return var_names

  def random_fn(self, sol, var_names):
    integer_var_names = []
    for var in var_names:
      if isinstance(self.mip.varname2var[var], IntegerVariable):
        integer_var_names.append(var)
    return integer_var_names

  def integral_fn(self, sol, var):
    mip = self.mip
    assert isinstance(mip.varname2var[var], IntegerVariable)
    partial_sol = sol.copy()
    # unfix current variable
    del partial_sol[var]
    submip = mip.fix(partial_sol, relax_integral_constraints=False)
    ass = scip_solve(submip)
    err = fabs(ass[var] - sol[var])
    return (err, var)

  def integral(self, sol, var_names, least_integral):
    int_vars = list(
        filter(lambda v: isinstance(self.mip.varname2var[v], IntegerVariable), var_names))
    errs = [
        self.integral_fn(sol, var)
        for var in np.random.choice(int_vars, min(2 * self.k, len(int_vars)), replace=False)
    ]
    if least_integral:
      # Furthest away.
      errs = sorted(errs, reverse=True)
    else:
      errs = sorted(errs)

    var_names = [var for err, var in errs[0:self.k]]
    # collect all variables with tie for the top-k
    for err, var_name in errs:
      if err == errs[self.k - 1][0]:
        if var_name not in var_names:
          var_names.append(var_name)
    return var_names

  def _heuristic_fn(self, i):
    var_names = self._heuristic(self._env.func_call_ith_env('get_curr_soln', i),
                                self._env.func_call_ith_env('get_varnames', i))
    act = []
    for vname in self.rng.choice(var_names, size=self.k, replace=False):
      act.append(self._env.func_call_ith_env('varname2idx', i, vname))
    return act

  def _scip_callback(self, _, sol, obj_val, step, heuristic_fn=None):
    ts = self._env.func_call_with_common_args('reset_solution', sol, obj_val)
    start_obj = np.min(ts.observation['curr_episode_log_values']['curr_obj'])
    assert math.fabs(obj_val - start_obj) <= EPS, (obj_val, start_obj, obj_val - start_obj)
    start_qual = np.min(ts.observation['curr_episode_log_values']['best_quality'])
    n_local_moves_start = ts.observation['n_local_moves']
    localmovetostats = {}
    localmovetostats[ts.observation['n_local_moves'][0]] = dict(
        obj=np.min(ts.observation['curr_episode_log_values']['curr_obj']),
        quality=np.min(ts.observation['curr_episode_log_values']['best_quality']),
        mip_work=np.mean(ts.observation['curr_episode_log_values']['mip_work']),
    )
    while np.all(
        ts.observation['n_local_moves'] - n_local_moves_start < self.config.n_local_moves):
      if heuristic_fn:
        action = np.asarray([heuristic_fn(i) for i in range(self.batch_size)], np.int32)
        for act in np.transpose(action):
          ts = self._env.step(act)
      else:
        step_output = self._shell.step(step_type=ts.step_type,
                                       reward=ts.reward,
                                       observation=ts.observation)
        ts = self._env.step(step_output.action)

      localmovetostats[ts.observation['n_local_moves'][0]] = dict(
          obj=np.min(ts.observation['curr_episode_log_values']['curr_obj']),
          quality=np.min(ts.observation['curr_episode_log_values']['best_quality']),
          mip_work=np.mean(ts.observation['curr_episode_log_values']['mip_work']),
      )
      # if the objective stalls for 3 or more consecutive local move steps break.
      if self._early_stop(localmovetostats): break
    # get the best improved solution among all the environments
    sol = self._env.func_call_ith_env(
        'get_curr_soln', np.argmin(ts.observation['curr_episode_log_values']['best_quality']))

    # stack all local moves together.
    stats = nest.map_structure(lambda *l: np.stack(l), *localmovetostats.values())
    final_qual = np.min(ts.observation['curr_episode_log_values']['best_quality'])
    final_obj = np.min(ts.observation['curr_episode_log_values']['curr_obj'])
    # solution improved.
    if final_obj < start_obj:
      print(f'Start objective: {start_obj} Final Objective: {final_obj}')
      print(f'Start Quality: {start_qual} Final Quality: {final_qual}')
      return sol, stats
    else:
      return None, stats

  def run_standalone(self, heuristic_fn):
    ts = self._env.reset()
    stats = dict(
        obj=np.min(ts.observation['curr_episode_log_values']['curr_obj']),
        quality=np.min(ts.observation['curr_episode_log_values']['best_quality']),
        mip_work=np.mean(ts.observation['curr_episode_log_values']['mip_work']),
    )
    localmovetostats = {0: stats}
    while np.all(ts.observation['n_local_moves'] < self.config.n_local_moves):
      if heuristic_fn:
        action = np.asarray([self._heuristic_fn(i) for i in range(self.batch_size)], np.int32)
        for act in np.transpose(action):
          ts = self._env.step(act)
      else:
        step_output = self._shell.step(step_type=ts.step_type,
                                       reward=ts.reward,
                                       observation=ts.observation)
        ts = self._env.step(step_output.action)
      localmovetostats[ts.observation['n_local_moves'][0]] = dict(
          obj=np.min(ts.observation['curr_episode_log_values']['curr_obj']),
          quality=np.min(ts.observation['curr_episode_log_values']['best_quality']),
          mip_work=np.mean(ts.observation['curr_episode_log_values']['mip_work']),
      )

    final_stats = nest.map_structure(lambda *l: np.stack(l), *localmovetostats.values())
    for k, l in final_stats.items():
      print(f'{k} improved from {l[0]} to {l[-1]}')
    with open(f'{self.config.results_dir}/out.pkl', 'wb') as f:
      pickle.dump(final_stats, f)
