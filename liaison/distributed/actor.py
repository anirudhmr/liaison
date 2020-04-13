"""Actor cls. Responsible for batched policy evaluation.
  Syncs the policy weights with PS and pushes the
  experience out to exp_sender.

  Env variables used:
    SYMPH_COLLECTOR_FRONTEND_HOST
    SYMPH_COLLECTOR_FRONTEND_PORT
    SYMPH_SPEC_PORT
"""

import logging
import os
from queue import Queue

import liaison.utils as U
from liaison.env.batch import ParallelBatchedEnv, SerialBatchedEnv
from liaison.utils import ConfigDict

from .exp_sender import ExpSender
from .full_episode_trajectory import Trajectory as FullEpisodeTrajectory
from .spec_server import SpecServer
from .trajectory import Trajectory


class Actor:
  """
  Actor is responsible for the following.

  (1) Create a shell and batched environments.
  (2) Pushes experience out to exp_sender.

  """

  def __init__(
      self,
      actor_id,
      shell_class,
      shell_config,
      env_class,
      env_configs,
      traj_length,
      seed,
      system_loggers,
      batch_size=1,  # num_envs
      n_unrolls=None,  # None => loop forever
      use_parallel_envs=False,
      use_threaded_envs=False,
      **sess_config):
    assert isinstance(actor_id, int)
    self.config = ConfigDict(sess_config)
    self.batch_size = batch_size
    self._traj_length = traj_length
    self._system_loggers = system_loggers
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
        seed=seed,
        batch_size=batch_size,
        **shell_config,
    )

    self._traj = Trajectory(obs_spec=self._obs_spec,
                            step_output_spec=self._shell.step_output_spec())

    if actor_id == 0:
      self._start_spec_server()

    self._setup_exp_sender()
    # blocking call -- runs forever
    self.run_loop(n_unrolls)

  def run_loop(self, n_unrolls):
    ts = self._env.reset()
    self._traj.reset()
    self._traj.start(next_state=self._shell.next_state, **dict(ts._asdict()))
    i = 0
    system_logs = {}
    while True:
      if n_unrolls is not None:
        if i == n_unrolls:
          return
      with U.Timer() as shell_step_timer:
        step_output = self._shell.step(step_type=ts.step_type,
                                       reward=ts.reward,
                                       observation=ts.observation)
      with U.Timer() as env_step_timer:
        ts = self._env.step(step_output.action)
      self._traj.add(step_output=step_output, **dict(ts._asdict()))
      if len(self._traj) == self._traj_length + 1:
        with U.Timer() as send_experience_timer:
          exps = self._traj.debatch_and_stack()
          self._traj.reset()
          self._send_experiences(exps)
          self._traj.start(next_state=self._shell.next_state, **dict(ts._asdict()))
        system_logs['put_experience_async_sec'] = send_experience_timer.to_seconds()

      for logger in self._system_loggers:
        logger.write(
            dict(shell_step_time_sec=shell_step_timer.to_seconds(),
                 env_step_time_sec=env_step_timer.to_seconds(),
                 **system_logs))
      i += 1

  def _setup_exp_sender(self):
    self._exp_sender = ExpSender(host=os.environ['SYMPH_COLLECTOR_FRONTEND_HOST'],
                                 port=os.environ['SYMPH_COLLECTOR_FRONTEND_PORT'],
                                 flush_iteration=None,
                                 manual_flush=True,
                                 compress_before_send=self.config.compress_before_send)

  def _send_experiences(self, exps):
    if hasattr(self, 'send_exp_queue'):
      q = self.send_exp_queue
    else:
      q = self.send_exp_queue = Queue(10)

      def f():
        while True:
          exps = q.get()
          for exp in exps:
            self._exp_sender.send(hash_dict=exp)
          self._exp_sender.flush()

      self.exp_thread = U.start_thread(f, daemon=True)
    q.put(exps)

  def _start_spec_server(self):
    logging.info("Starting spec server.")
    print("Starting spec server.")
    self._spec_server = SpecServer(port=os.environ['SYMPH_SPEC_PORT'],
                                   traj_spec=self._traj.spec,
                                   action_spec=self._action_spec)
    self._spec_server.start()
