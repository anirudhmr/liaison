"""Define trajectory class to keep track of what's being shipped to replay."""

from __future__ import absolute_import, division, print_function

import copy
import functools

import numpy as np
from liaison.agents import StepOutput
from liaison.distributed.trajectory import Trajectory as BaseTrajectory
from liaison.env import StepType
from liaison.specs import ArraySpec, BoundedArraySpec
from tensorflow.contrib.framework import nest


def expand_spec(spec):
  spec = copy.deepcopy(spec)
  spec.expand_dims(None, axis=0)
  return spec


class Trajectory(BaseTrajectory):
  """
  Needs to collect the step environment outputs and
  agent outputs and stack them up for replay storage.

  Must implement the following functions:
    add: Adds a timestep to the trajectory
    stack: Stacks the trajectories to get a leading time dimension and
    converts attributes to a dict of serializable types (np.array, list, etc.)
    reset: Clear the existing traj info.
  """

  def __init__(self, obs_spec, step_output_spec, batch_size, discount_factor,
               traj_length):
    self._batch_size = batch_size
    self._traj_len = traj_length
    self._discount_factor = discount_factor
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
    # self._trajs[i] = trajectory of the ith experience in the batch.
    self._trajs = None
    # list of timesteps that have been backtracked
    # and ready to be split into chunks to be shipped out.
    # Note that this means that we assume it is natural transition from
    # the end of an episode of one environment to the begin of episode
    # of another environment. This assumption might not hold true if there
    # is state maintained anywhere in the timesteps/agent actions
    # that is carried across the episodes of any given environment.
    self._finished_timesteps = []
    # used to chop the trajectory into chunks.
    self._chopping_traj = BaseTrajectory(obs_spec, step_output_spec)

  def reset(self):
    self._trajs = [[] for _ in range(self._batch_size)]
    self._chopping_traj.reset()

  def start(self, step_type, reward, discount, observation, next_state):
    self.add(step_type, reward, discount, observation,
             StepOutput(next_state=next_state, action=None, logits=None))

  def add(self, step_type, reward, discount, observation, step_output):

    ts = dict(step_type=step_type,
              reward=reward,
              discount=discount,
              observation=observation,
              step_output=step_output._asdict())

    for i, ts in enumerate(self.debatch_timestep(ts)):
      while len(self._trajs) <= i:
        self._trajs.append([])
      self._trajs[i].append(ts)
      if ts['step_type'] == StepType.LAST:
        self._backtrack_trajectory(self._trajs[i])
        self._trajs[i] = []

  def _backtrack_trajectory(self, traj):
    """
      traj: List of timesteps that are ready to be backtracked.
    """
    assert traj[0]['step_type'] == StepType.FIRST
    assert all([ts['step_type'] == StepType.MID for ts in traj[1:-1]])
    assert traj[-1]['step_type'] == StepType.LAST
    value = 0
    vals = []
    for ts in reverse(traj):
      value = ts['discount'] * ts['reward'] + self._disc_factor * value
      vals.append(value)
    vals = list(reverse(vals))
    # shift the vals by one to the left by removing head and appending 0
    vals.pop(0)
    vals.append(0)
    for ts, val in zip(traj, vals):
      ts['value_boostrap'] = val
    self._finished_timesteps.extend(traj)

  def debatch_timestep(self, ts):
    """Debatches a single timestep.
    Returns bs length of timesteps."""

    traj_spec = self._traj_spec

    def f(arr):
      if arr is None:
        return arr
      l = np.split(arr, len(arr))
      # remove the leading dimension
      l = list(map(functools.partial(np.squeeze, axis=0), l))
      return l

    # split along the batch dimension
    d = nest.map_structure_up_to(traj_spec, f, ts)

    # determine the batch size
    lens = [
        len(v) for v in filter(lambda k: k is not None,
                               nest.flatten_up_to(traj_spec, d))
    ]
    bs = lens[0]
    assert all(x == bs for x in lens)

    # Flatten and replicate by packing the sequence bs times.
    d = nest.flatten_up_to(traj_spec, d)

    l = []
    for i in range(bs):
      l.append(
          nest.pack_sequence_as(
              traj_spec, list(map(lambda k: k if k is None else k[i], d))))
    return l

  def debatch_and_stack(self):
    traj_len = self._traj_len
    chopping_traj = self._chopping_traj

    exps = []
    for ts in self._finished_timesteps:
      if len(chopping_traj) == 0:
        # tihs branch is taken only after reset is called on trajectory.
        chopping_traj.start(
            next_state=ts['step_output']['next_state'],
            # remove step_output from ts
            **{k: v
               for k, v in ts.items() if k != 'step_output'})
        assert ts['step_output']['action'] is None
        assert ts['step_output']['logits'] is None
        continue

      chopping_traj.add(**ts)
      assert len(chopping_traj) <= traj_len + 1

      if len(chopping_traj) == traj_len + 1:
        exps.extend(chopping_traj.debatch_and_stack())
        chopping_traj.reset()

      if len(chopping_traj) == 0:
        chopping_traj.start(
            next_state=ts['step_output']['next_state'],
            # remove step_output from ts
            **{k: v
               for k, v in ts.items() if k != 'step_output'})

    return exps
