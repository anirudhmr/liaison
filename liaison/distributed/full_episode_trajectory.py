"""Define trajectory class to keep track of what's being shipped to replay."""

import copy
import functools

import numpy as np
from liaison.agents import StepOutput
from liaison.distributed.trajectory import Trajectory as BaseTrajectory
from liaison.env import StepType
from liaison.specs import ArraySpec, BoundedArraySpec
from liaison.utils import ConfigDict
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
    # _finished_timesteps[i] = Finished timesteps for the ith item of the batch.
    self._finished_timesteps = None
    # used to chop the trajectory into chunks.

    obs_spec2 = copy.deepcopy(obs_spec)
    obs_spec2['bootstrap_value'] = ArraySpec(dtype=np.float32,
                                             shape=(None, ),
                                             name='bootstrap_value_spec')
    self._chopping_trajs = [
        BaseTrajectory(obs_spec2, step_output_spec) for _ in range(batch_size)
    ]
    self._len = 0

  def reset(self):
    self._trajs = [[] for _ in range(self._batch_size)]
    for traj in self._chopping_trajs:
      traj.reset()
    self._len = 0
    self._finished_timesteps = [[] for _ in range(self._batch_size)]

  def start(self, step_type, reward, discount, observation, next_state):
    self.add(step_type, reward, discount, observation,
             StepOutput(next_state=next_state, action=None, logits=None))

  def add(self, step_type, reward, discount, observation, step_output):

    self._len += 1
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
        self._backtrack_trajectory(i, self._trajs[i])
        self._trajs[i] = []

  def _backtrack_trajectory(self, i, traj):
    """
      traj: List of timesteps that are ready to be backtracked.
    """
    assert traj[0]['step_type'] == StepType.FIRST
    assert all([ts['step_type'] == StepType.MID for ts in traj[1:-1]])
    assert traj[-1]['step_type'] == StepType.LAST
    value = 0.0
    vals = []
    for ts in reversed(traj):
      value = ts['reward'] + ts['discount'] * self._discount_factor * value
      vals.append(value)
    vals = list(reversed(vals))
    # shift the vals by one to the left by removing head and appending 0
    vals.pop(0)
    vals.append(0)
    for ts, val in zip(traj, vals):
      ts['observation']['bootstrap_value'] = val
    self._finished_timesteps[i].extend(traj)

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

  def __len__(self):
    return self._len

  def debatch_and_stack(self):
    traj_len = self._traj_len

    exps = []
    for i, finished_ts in enumerate(self._finished_timesteps):
      chopping_traj = self._chopping_trajs[i]
      for ts in finished_ts:
        if len(chopping_traj) == 0:
          # tihs branch is taken only after reset is called on trajectory.
          chopping_traj.start(
              next_state=ts['step_output']['next_state'],
              # remove step_output from ts
              **ConfigDict(
                  **{k: v
                     for k, v in ts.items() if k != 'step_output'}))
          assert ts['step_output']['action'] is None
          assert ts['step_output']['logits'] is None
          continue

        chopping_traj.add(**ConfigDict(**ts))
        assert ts['step_output']['action'] is not None
        assert len(chopping_traj) <= traj_len + 1

        if len(chopping_traj) == traj_len + 1:
          # TODO: Add dummy batch dimension and use debatch_and_stack
          # for uniformity.
          exps.append(chopping_traj.stack())
          chopping_traj.reset()
          chopping_traj.start(
              next_state=ts['step_output']['next_state'],
              # remove step_output from ts
              **ConfigDict(
                  **{k: v
                     for k, v in ts.items() if k != 'step_output'}))

      self._finished_timesteps[i] = []

    def f(path, spec, v):
      if path[0] == 'step_output' and path[1] != 'next_state':
        assert len(v) == traj_len
        return
      assert len(v) == traj_len + 1

    assert all([
        nest.map_structure_with_tuple_paths_up_to(self.spec, f, self.spec, exp)
        for exp in exps
    ])
    return exps

  @property
  def spec(self):
    return self._chopping_trajs[0].spec
