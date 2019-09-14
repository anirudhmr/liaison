"""Define trajectory class to keep track of what's being shipped to replay."""

from __future__ import absolute_import, division, print_function

import copy
import numpy as np
from liaison.agents import StepOutput
from liaison.specs import ArraySpec, BoundedArraySpec
from tensorflow.contrib.framework import nest


def expand_spec(spec):
  spec = copy.deepcopy(spec)
  spec.expand_dims(None, axis=0)
  return spec


class Trajectory(object):
  """
  Needs to collect the step environment outputs and
  agent outputs and stack them up for replay storage.

  Must implement the following functions:
    add: Adds a timestep to the trajectory
    stack: Stacks the trajectories to get a leading time dimension and
    converts attributes to a dict of serializable types (np.array, list, etc.)
    reset: Clear the existing traj info.
  """

  def __init__(
      self,
      obs_spec,
      action_spec,
      step_output_spec,
  ):
    self._traj = None
    # per step action spec so not expanding its shape.
    self._action_spec = action_spec
    # Don't use shape in the spec since it's unknown
    self._traj_spec = dict(
        step_type=ArraySpec(dtype=np.int8,
                            shape=(None, None),
                            name='traj_step_type_spec'),
        reward=ArraySpec(dtype=np.float32,
                         shape=(None, None),
                         name='traj_reward_spec'),
        discount=ArraySpec(dtype=np.float32,
                           shape=(None, None),
                           name='traj_discount_spec'),
        observation=nest.map_structure(expand_spec, obs_spec),
        step_output=nest.map_structure(expand_spec, step_output_spec))

  def start(self, step_type, reward, discount, observation, next_state):
    self.add(step_type, reward, discount, observation,
             StepOutput(next_state=next_state, action=None, logits=None))

  def add(self, step_type, reward, discount, observation, step_output):

    traj = dict(step_type=step_type,
                reward=reward,
                discount=discount,
                observation=observation,
                step_output=step_output._asdict())
    self._traj.append(traj)

  def reset(self):
    self._traj = []

  @property
  def spec(self):
    return self._traj_spec

  def stack(self):
    stacked_trajs = []

    def f(spec, *l):
      l = list(filter(lambda k: k is not None, l))
      return np.stack(l, axis=0).astype(spec.dtype)

    stacked_trajs = nest.map_structure_up_to(self._traj_spec, f,
                                             self._traj_spec, *self._traj)
    return stacked_trajs

  def __len__(self):
    if self._traj:
      return len(self._traj)
    else:
      return 0

  @staticmethod
  def format_traj_spec(self, traj_spec, bs, traj_length):
    """Fills in the missing shape fields of the traj spec."""

    # All keys starting with following should get traj_length first dimension
    t_plus_one = [
        'step_type', 'reward', 'discount', 'observation',
        'step_output/next_state'
    ]
    t = ['step_output/action', 'step_output/logits']

    def f(path, v):
      if any([path.startswith(v, x) for x in t_plus_one]):
        v.set_shape((traj_length + 1, bs) + v.shape[2:])
      else:
        v.set_shape((traj_length, bs) + v.shape[2:])

      return v

    traj_spec = nest.map_structure(f, traj_spec)
    return traj_spec
