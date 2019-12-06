"""
  Evaluator cls. Responsible for batched policy evaluation.
  Returns mean and variance of multiple parallel random environments.
"""
import time

import numpy as np

import tree as nest
from liaison.env import StepType
from liaison.env.batch import ParallelBatchedEnv, SerialBatchedEnv


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
      seed,
      max_evaluations=int(1e6),
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
    self._shell = shell_class(
        action_spec=self._action_spec,
        obs_spec=self._obs_spec,
        batch_size=batch_size,
        seed=seed,
        **shell_config,
    )

    # blocking call -- runs forever
    self._run_loop()

  def _run_loop(self):
    # performs n_evaluations before exiting
    for i in range(self.max_evaluations):
      eval_log_values = []
      for i in range(self._n_trials):
        # perform an evaluation
        # env_mask[i] = should the ith env be masked for the shell.
        env_mask = np.ones(self.batch_size, dtype=bool)

        ts = self._env.reset()
        ep_rew = ts.reward
        log_values = [None] * len(env_mask)
        while np.any(env_mask):
          for i, mask in enumerate(env_mask):
            if mask and ts.step_type[i] == StepType.LAST:
              env_mask[i] = False
              log_values[i] = dict(ep_reward=ep_rew[i])
              if 'log_values' in ts.observation:
                # just use the last timestep's log_values.
                for k, v in ts.observation['log_values'].items():
                  log_values[i].update({f'log_values/{k}': v[i]})

          step_output = self._shell.step(step_type=ts.step_type,
                                         reward=ts.reward,
                                         observation=ts.observation)
          ts = self._env.step(step_output.action)
          ep_rew += (ts.reward * env_mask)
        eval_log_values.append(log_values)

      # stack all log_values
      log_values = nest.map_structure(lambda *l: np.vstack(l),
                                      *eval_log_values)
      for logger in self._loggers:
        logger.write(log_values)

      if i != self.max_evaluations - 1:
        time.sleep(self._eval_sleep_time)
    print('Evaluator done!!')
