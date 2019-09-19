"""Batch several copies of environment together."""

from __future__ import absolute_import, division, print_function

import numpy as np
from liaison.env import Env, TimeStep
from liaison.specs import ArraySpec, BoundedArraySpec
from tensorflow.contrib.framework import nest


def stack_specs(*specs):
  for spec in specs[1:]:
    assert spec.shape == specs[0].shape
    assert spec.dtype == specs[0].dtype
    assert spec.name == specs[0].name

  if isinstance(specs[0], BoundedArraySpec):
    min_min = specs[0].minimum
    max_max = specs[0].maximum
    for spec in specs:
      min_min = min(min_min, spec.minimum)
      max_max = max(max_max, spec.maximum)

    spec = BoundedArraySpec(
        (len(specs), ) + specs[0].shape,
        specs[0].dtype,
        min_min,
        max_max,
        name='batched_' + ('spec' if specs[0].name is None else specs[0].name))
  else:
    specs[0].expand_dims(len(specs), axis=0)
    spec = specs[0]

  return spec


class BatchedEnv(Env):

  def __init__(self, n_envs, env_class, env_configs, seed):
    assert n_envs >= 1
    assert len(env_configs) == n_envs
    self._n_envs = n_envs
    self._env_configs = env_configs
    self._env_class = env_class
    self.seed = seed

  def _stack_specs(self, specs):
    return nest.map_structure(stack_specs, *specs)

  def _make_traj_spec(self, obs_spec):
    self._traj_spec = dict(step_type=BoundedArraySpec(dtype=np.int8,
                                                      shape=(),
                                                      minimum=0,
                                                      maximum=2,
                                                      name='batched_env_step'
                                                      '_type_spec'),
                           reward=ArraySpec(dtype=np.float32,
                                            shape=(),
                                            name='batched_env_reward_spec'),
                           discount=ArraySpec(
                               dtype=np.float32,
                               shape=(),
                               name='batched_env_discount_spec'),
                           observation=obs_spec)
    return self._traj_spec

  def observation_spec(self):
    return self._obs_spec

  def action_spec(self):
    return self._action_spec

  def _stack_ts(self, timesteps):
    """Should be called after _make_traj_spec."""

    dict_tss = []
    for ts in timesteps:
      dict_tss.append(dict(ts._asdict()))

    def f(spec, *l):
      return np.stack(l, axis=0).astype(spec.dtype)

    stacked_ts = nest.map_structure_up_to(self._traj_spec, f, self._traj_spec,
                                          *dict_tss)
    return TimeStep(**stacked_ts)
