"""
  Evaluator cls. Responsible for batched policy evaluation.
  Returns mean and variance of multiple parallel random environments.
"""
import copy
import sys
import time
from multiprocessing.pool import ThreadPool
from threading import Thread

import numpy as np

import tree as nest
from liaison.daper.milp.heuristics.heuristic_fn import run as heuristic_run
from liaison.env import StepType
from liaison.env.batch import ParallelBatchedEnv, SerialBatchedEnv
from liaison.env.rins import Env


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
      env_class,
      env_configs,
      traj_length,
      loggers,
      heuristic_loggers,
      seed,
      max_evaluations=int(1e6),
      n_trials=2,
      eval_sleep_time=15 * 60,
      batch_size=1,  # num_envs
      use_parallel_envs=False,
      use_threaded_envs=False,
      **unused_config):
    del unused_config
    self.batch_size = batch_size
    self._traj_length = traj_length
    self._loggers = loggers
    self.max_evaluations = max_evaluations
    self._n_trials = n_trials
    self._eval_sleep_time = eval_sleep_time
    if use_parallel_envs:
      self._env = ParallelBatchedEnv(batch_size,
                                     env_class,
                                     env_configs,
                                     seed,
                                     use_threads=use_threaded_envs)
    else:
      self._env = SerialBatchedEnv(batch_size, env_class, env_configs, seed)
    self._action_spec = self._env.action_spec()
    self._obs_spec = self._env.observation_spec()

    if 'sync_period' in shell_config:
      del shell_config['sync_period']

    self._shell = shell_class(
        action_spec=self._action_spec,
        obs_spec=self._obs_spec,
        batch_size=batch_size,
        seed=seed,
        sync_period=int(1e20),  # don't sync unless asked to do so explicitly.
        **shell_config,
    )

    t = Thread(target=self._collect_heuristics,
               args=(env_class, env_configs, seed, heuristic_loggers))
    t.start()
    # blocking call -- runs forever
    self._run_loop()
    t.join()

  def _collect_heuristics(self, env_class, env_configs, seed,
                          heuristic_loggers):

    def f(i):
      """Run in parallel process."""
      config = copy.deepcopy(env_configs[i])
      config.update(lp_features=False, seed=seed, make_obs_for_mlp=False)
      env = env_class(id=i, **config)
      return heuristic_run(
          'random', env.k, self._n_trials,
          [seed + i * self._n_trials + j for j in range(self._n_trials)], env)

    with ThreadPool(8) as pool:
      l = pool.map(f, list(range(len(env_configs))))

    for i in range(len(l)):
      for j in range(len(l[i])):
        # stack the timesteps first
        l[i][j] = nest.map_structure(lambda *l: np.stack(l), *l[i][j])
      # now stack the  trials
      l[i] = nest.map_structure(lambda *l: np.stack(l), *l[i])

    # final output is (num_trials, num_envs, n_local_moves)
    log_values = nest.map_structure(lambda *l: np.swapaxes(np.stack(l), 0, 1),
                                    *l)
    for logger in heuristic_loggers:
      logger.write(log_values)

  def _run_loop(self):
    # performs n_evaluations before exiting
    for eval_id in range(self.max_evaluations):
      self._shell.sync()
      print(f'Starting evaluation {eval_id}')
      # eval_log_values -> List[List[Dict[str, numeric]]]
      # eval_log_values[i] = eval log values for the ith trial
      # eval_log_values[.][j] -> log values for the jth environment
      eval_log_values = []
      # each evaluation is repeated for n_trials
      for trial_id in range(self._n_trials):
        print(f'Starting trial {trial_id}')

        # perform an evaluation
        # env_mask[i] = should the ith env be masked for the shell.
        env_mask = np.ones(self.batch_size, dtype=bool)

        ts = self._env.reset()
        log_values = [[] for _ in range(len(env_mask))]
        first_step = True

        while np.any(env_mask):
          obs = ts.observation
          for i, mask in enumerate(env_mask):
            if 'graph_features' in obs:
              globals_ = obs['graph_features']['globals']
            else:
              globals_ = obs['globals']

            if mask and (globals_[i][Env.GLOBAL_LOCAL_SEARCH_STEP]
                         or first_step):
              d = nest.map_structure(lambda v: v[i],
                                     obs['curr_episode_log_values'])
              d.update(
                  rew=ts.reward[i],
                  optimal_solution=obs['optimal_solution'][i],
                  optimal_lp_solution=obs['optimal_lp_solution'][i],
                  current_solution=obs['current_solution'][i],
              )
              log_values[i].append(d)
              first_step = False

          for i, mask in enumerate(env_mask):
            if mask and ts.step_type[i] == StepType.LAST:
              env_mask[i] = False

          step_output = self._shell.step(step_type=ts.step_type,
                                         reward=ts.reward,
                                         observation=ts.observation)
          ts = self._env.step(step_output.action)

        for i, _ in enumerate(log_values):
          log_values[i] = nest.map_structure(lambda *l: np.stack(l),
                                             *log_values[i])
        eval_log_values.append(
            nest.map_structure(lambda *l: np.stack(l), *log_values))

      # log_values[i] -> For the ith environment log values where each value
      # has a dimension (n_trials, n_envs) + added at its forefront.
      log_values = nest.map_structure(lambda *l: np.stack(l), *eval_log_values)
      for logger in self._loggers:
        logger.write(log_values)

      if i != self.max_evaluations - 1:
        time.sleep(self._eval_sleep_time)

    print('Evaluator is done!!')
