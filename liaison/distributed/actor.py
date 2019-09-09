"""Actor cls. Responsible for batched policy evaluation. Syncs the policy weights with PS and pushes the experience out to exp_sender."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from env import SerialBatchedEnv
from distributed import Trajectory


class Actor:
  """
  Actor is responsible for the following.

  (1) Create a shell and batched environments.
  (2) Pushes experience out to exp_sender.

  """

  def __init__(
      self,
      shell_class,
      shell_config,
      env_class,
      env_configs,
      traj_length,
      seed,
      exp_sender_handle,
      batch_size=1,  # num_envs
      n_unrolls=None,  # None => loop forever
      **kwargs):
    del kwargs
    self._traj_length = traj_length
    self._exp_sender_handle = exp_sender_handle
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
    self.run_loop(n_unrolls)

  def run_loop(self, n_unrolls):
    ts = self._env.reset()
    self._traj.reset()
    self._traj.start(next_state=self._shell.next_state, **dict(ts._asdict()))
    i = 0
    while True:
      if n_unrolls is not None:
        if i == n_unrolls:
          return
      step_output = self._shell.step(step_type=ts.step_type,
                                     reward=ts.reward,
                                     observation=ts.observation)
      ts = self._env.step(step_output.action)
      self._traj.add(step_output=step_output, **dict(ts._asdict()))
      if len(self._traj) == self._traj_length:
        exp = self._traj.stack_and_flatten()
        self._send_experience(exp)
        self._traj.reset()
        self._traj.start(next_state=self._shell.next_state,
                         **dict(ts._asdict()))
      i += 1

  def _send_experience(self, exp):
    self._exp_sender_handle.send(hash_dict=exp)
