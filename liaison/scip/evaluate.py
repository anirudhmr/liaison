# Evaluates the agent inside SCIP
import copy
import pickle
import sys
import time
from multiprocessing.pool import ThreadPool
from threading import Thread

import numpy as np

import tree as nest
from liaison.daper.milp.heuristics.heuristic_fn import run as heuristic_run
from liaison.env import StepType
from liaison.env.batch import ParallelBatchedEnv, SerialBatchedEnv
from liaison.env.utils.rins import get_sample
from liaison.scip.scip_integration import (EvalHeur, init_scip_params,
                                           run_branch_and_bound_scip)
from liaison.utils import ConfigDict
from pyscipopt import Model


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
      **config):
    self.config = ConfigDict(config)
    # get the mip instance
    milp = get_sample(dataset, dataset_type, graph_start_idx)
    # init scip model
    self._model = model = Model()
    model.setRealParam('limits/gap', args.gap)
    model.hideOutput()
    milp.mip.add_to_scip_solver(model)
    init_scip_params(model, seed=seed)
    model.presolve()

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

    if use_parallel_envs:
      self._env = ParallelBatchedEnv(batch_size,
                                     env_class,
                                     env_configs,
                                     seed,
                                     use_threads=use_threaded_envs)
    else:
      self._env = SerialBatchedEnv(batch_size, env_class, env_configs, seed)

    # stop syncing
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

  def run(self):
    heuristic = EvalHeur(self._primal_heuristic_callback)
    results = run_branch_and_bound_scip(self._model, heuristic)
    log_vals = heuristic._obj_vals
    with open('/data/nms/tfp/dump.pkl', 'wb') as f:
      pickle.dump(log_vals, f)

  def _primal_heuristic_callback(self, model, sol, obj_val, step):
    ts = self._env.func_call_with_common_args('reset_solution', sol, obj_val)

    start_obj = np.min(ts.observation['curr_episode_log_values']['curr_obj'])
    start_qual = np.min(
        ts.observation['curr_episode_log_values']['best_quality'])
    n_local_moves_start = ts.observation['n_local_moves']
    stats = dict(mip_work=0)
    while ts.observation[
        'n_local_moves'] - n_local_moves_start <= self.config.n_local_moves:
      step_output = self._shell.step(step_type=ts.step_type,
                                     reward=ts.reward,
                                     observation=ts.observation)
      ts = self._env.step(step_output.action)
      stats['mip_work'] += ts.observation['curr_episode_log_values'][
          'mip_work']

    # get the best improved solution among all the environments
    sol = self._env.func_call_ith_env(
        'get_curr_soln',
        np.argmin(ts.observation['curr_episode_log_values']['best_quality']))

    final_qual = np.min(
        ts.observation['curr_episode_log_values']['best_quality'])
    final_obj = np.min(ts.observation['curr_episode_log_values']['curr_obj'])

    # solution improved.
    if final_obj < start_obj:
      print(f'Start objective: {start_obj} Final Objective: {final_obj}')
      print(f'Start Quality: {start_qual} Final Quality: {final_qual}')
      return sol, stats
    else:
      return None, stats
