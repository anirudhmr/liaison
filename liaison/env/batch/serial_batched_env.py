"""Batch several copies of environment together."""

from __future__ import absolute_import, division, print_function

from liaison.env.batch import BaseBatchedEnv


class BatchedEnv(BaseBatchedEnv):

  def __init__(self, n_envs, env_class, env_configs, seed, **kwargs):

    super(BatchedEnv, self).__init__(n_envs, env_class, env_configs, seed)
    self._envs = []
    for i in range(n_envs):
      env = env_class(id=i, seed=seed, **env_configs[i])
      self._envs.append(env)

    obs_specs = [env.observation_spec() for env in self._envs]
    self._obs_spec = self._stack_specs(obs_specs)
    action_specs = [env.action_spec() for env in self._envs]
    self._action_spec = self._stack_specs(action_specs)

    self._make_step_spec(self._obs_spec)
    self.set_seeds(seed)

  def reset(self):
    timesteps = []
    for env in self._envs:
      ts = env.reset()
      timesteps.append(ts)

    return self._stack_ts(timesteps)

  def step(self, action):
    timesteps = []
    for i, env in enumerate(self._envs):
      ts = env.step(action[i])
      timesteps.append(ts)

    return self._stack_ts(timesteps)

  def set_seeds(self, seed):
    for env in self._envs:
      env.set_seed(seed)

  def func_call_with_common_args(self, f_str: str, *args, **kwargs):
    timesteps = []
    for env in self._envs:
      ts = getattr(env, f_str)(*args, **kwargs)
      timesteps.append(ts)
    return self._stack_ts(timesteps)

  def func_call_ith_env(self, func_name, i, *args, **kwargs):
    # gets the attr_name of the ith environment
    return getattr(self._envs[i], func_name)(*args, **kwargs)
