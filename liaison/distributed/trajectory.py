"""Define trajectory class to keep track of what's being shipped to replay."""

from __future__ import absolute_import, division, print_function

import functools
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
      step_output_spec,
  ):
    self._trajs = None
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
    self._trajs.append(traj)

  def reset(self):
    self._trajs = []

  @property
  def spec(self):
    return self._traj_spec

  @staticmethod
  def _stack(trajs, traj_spec):
    stacked_trajs = []

    def f(spec, *l):
      l = list(filter(lambda k: k is not None, l))
      return np.stack(l, axis=0).astype(spec.dtype)

    return nest.map_structure_up_to(traj_spec, f, traj_spec, *trajs)

  def stack(self):
    return Trajectory._stack(self._trajs, self._traj_spec)

  def debatch_and_stack(self):
    """Remove the leading batch dimension and then stack.
    Returns list of stacked timesteps."""
    traj_spec = self._traj_spec

    def f(arr):
      return None if arr is None else np.split(arr, len(arr))

    l = []
    for traj in self._trajs:
      # split along the batch dimension
      d = nest.map_structure_up_to(traj_spec, f, traj)

      # determine the batch size
      lens = [
          len(v) for _, v in filter(lambda k: k is not None,
                                    nest.flatten_up_to(traj_spec, d))
      ]
      bs = lens[0]
      assert all(x == bs for x in lens)

      # Flatten and replicate by packing the sequence bs times.
      d = nest.flatten_up_to(traj_spec, d)
      if not l:
        l = [[] for _ in range(bs)]

      for i in range(bs):
        l[i].append(
            nest.pack_sequence_as(
                traj_spec, list(map(lambda k: None
                                    if k is None else k[i], d))))

    return list(
        map(functools.partial(Trajectory._stack, traj_spec=self._traj_spec),
            l))

  @staticmethod
  def batch(self, trajs, traj_spec):
    batched_trajs = Trajectory._stack(trajs, traj_spec)

    def f(spec, l):
      return None if l is None else np.swapaxes(l, 0, 1)

    return nest.map_structure_up_to(traj_spec, f, batched_trajs)

  def __len__(self):
    if self._trajs:
      return len(self._trajs)
    else:
      return 0

  @staticmethod
  def format_traj_spec(traj_spec, bs, traj_length):
    """Fills in the missing shape fields of the traj spec."""

    # All keys starting with following should get traj_length first dimension
    t_plus_one = [
        'step_type', 'reward', 'discount', 'observation',
        'step_output/next_state'
    ]
    t = ['step_output/action', 'step_output/logits']

    def f(path, v):
      if any([path.startswith(x) for x in t_plus_one]):
        v.set_shape((traj_length + 1, bs) + v.shape[2:])
      else:
        v.set_shape((traj_length, bs) + v.shape[2:])

      return v

    traj_spec = nest.map_structure_with_paths(f, traj_spec)
    return traj_spec
