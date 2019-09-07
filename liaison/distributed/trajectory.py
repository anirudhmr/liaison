"""Define trajectory class to keep track of what's being shipped to replay."""

from __future__ import absolute_import, division, print_function

import numpy as np
from agents import StepOutput
from specs import ArraySpec, BoundedArraySpec
from tensorflow.contrib.framework import nest


class Trajectory:
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
    self._action_spec = action_spec
    # Don't use shape in the spec since it's unknown
    self._traj_spec = dict(step_type=BoundedArraySpec(dtype=np.int8,
                                                      shape=(),
                                                      minimum=0,
                                                      maximum=2,
                                                      name='traj_step'
                                                      '_type_spec'),
                           reward=ArraySpec(dtype=np.float32,
                                            shape=(),
                                            name='traj_reward_spec'),
                           discount=ArraySpec(dtype=np.float32,
                                              shape=(),
                                              name='traj_discount_spec'),
                           observation=obs_spec,
                           step_output=step_output_spec)

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

  def stack_and_flatten(self):
    stacked_trajs = []

    def f(spec, *l):
      l = list(filter(lambda k: k is not None, l))
      return np.stack(l, axis=0).astype(spec.dtype)

    stacked_trajs = nest.map_structure_up_to(self._traj_spec, f,
                                             self._traj_spec, *self._traj)
    flattened_trajs = nest.flatten_up_to(self._traj_spec, stacked_trajs)
    return flattened_trajs

  def __len__(self):
    if self._traj:
      return len(self._traj)
    else:
      return 0
