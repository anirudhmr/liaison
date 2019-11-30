"""
  Evaluator cls. Responsible for batched policy evaluation.
  Returns mean and variance of multiple parallel random environments.
"""
import numpy as np

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
      batch_size=1,  # num_envs
      use_parallel_envs=False,
      use_threaded_envs=False,
      **unused_config):

    del unused_config
    self.batch_size = batch_size
    self._traj_length = traj_length
    self._loggers = loggers
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
    ts = self._env.reset()
    ep_rew = ts.reward
    for i in range(self._traj_length):
      step_output = self._shell.step(step_type=ts.step_type,
                                     reward=ts.reward,
                                     observation=ts.observation)
      ts = self._env.step(step_output.action)
      ep_rew += ts.reward

    log_values = dict(ep_reward_mean=np.mean(ep_rew),
                      ep_reward_std=np.std(ep_rew))

    if 'log_values' in ts.observation:
      # just use the last timestep's log_values.
      for k, v in ts.observation['log_values'].items():
        log_values['log_values/%s_mean' % k] = np.mean(v)
        log_values['log_values/%s_std' % k] = np.std(v)

    # send log_values
    for logger in self._loggers:
      logger.write(log_values)
