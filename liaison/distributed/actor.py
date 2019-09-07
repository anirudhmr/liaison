"""Actor cls. Responsible for batched policy evaluation. Syncs the policy weights with PS and pushes the experience out to replay."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from env import SerialBatchedEnv
from distributed import Trajectory


class Actor:
  """
  Actor is responsible for the following.

  (1) Create a shell and batched environments.
  (2) Pushes experience out to replay.

  """

  def __init__(
      self,
      shell_class,
      shell_config,
      env_class,
      env_configs,
      traj_length,
      seed,
      replay_handle,
      batch_size=1,  # num_envs
      **kwargs):
    del kwargs
    self._traj_length = traj_length
    self._replay_handle = replay_handle
    self._env = SerialBatchedEnv(batch_size, env_class, env_configs, seed)
    self._action_spec = self._env.action_spec()
    self._obs_spec = self._env.observation_spec()

    self._shell = shell_class(
        action_spec=self._action_spec,
        obs_spec=self._obs_spec,
        batch_size=batch_size,
        **shell_config,
    )

    self._traj = Trajectory(obs_spec=self._obs_spec,
                            action_spec=self._action_spec,
                            step_output_spec=self._shell.step_output_spec())

    # blocking call -- runs forever
    self.run_loop()

  def run_loop(self):
    ts = self._env.reset()
    self._traj.reset()
    self._traj.start(next_state=self._shell.next_state, **dict(ts._asdict()))
    while True:
      step_output = self._shell.step(step_type=ts.step_type,
                                     reward=ts.reward,
                                     observation=ts.observation)
      ts = self._env.step(step_output.action)
      self._traj.add(step_output=step_output, **dict(ts._asdict()))
      if len(self._traj) == self._traj_length:
        exp = self._traj.stack_and_flatten()
        self._traj.reset()
        self._push_replay(exp)
        self._traj.start(next_state=self._shell.next_state,
                         **dict(ts._asdict()))

  def _push_replay(self, exp):
    self._replay_handle.push(exp)
