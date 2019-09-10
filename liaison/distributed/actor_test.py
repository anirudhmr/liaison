"""Test file for ur discrete."""

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from liaison.agents import BaseAgent, StepOutput
from liaison.distributed import Actor, Shell
from liaison.env import StepType, TimeStep
from liaison.env.xor_env import XOREnv
from liaison.specs.specs import ArraySpec, BoundedArraySpec

B = 8


class DummyAgent(BaseAgent):

  def __init__(self, name, **kwargs):
    del kwargs
    self._name = name

  def _dummy_state(self, batch_size):
    return tf.fill(tf.expand_dims(batch_size, 0), 0)

  def initial_state(self, batch_size):
    return self._dummy_state(batch_size)

  def step(self, step_type, reward, obs, prev_state):
    """Pick a random discrete action from action_spec."""
    with tf.variable_scope(self._name):
      with tf.name_scope('ur_step'):
        batch_size = tf.shape(step_type)[0]
        action = tf.fill((batch_size, ), 0)
        logits = tf.fill(tf.expand_dims(batch_size, 0), 0)
        return StepOutput(action, logits, self._dummy_state(batch_size))


class DummyPS(object):
  count = 0

  def pull(self, var_names):
    DummyPS.count += 1
    return {var_name: 0 for var_name in var_names}


class DummyReplay:

  def send(self, *args, **kwargs):
    pass


TRAJ_LENGTH = 10
N_ENVS = 2
SEED = 42


class ActorTest(tf.test.TestCase):

  def _get_actor(self):

    shell_config = dict(
        agent_class=DummyAgent,
        agent_config={},
        sync_period=5,
        ps_handle=DummyPS(),
    )
    return Actor(
        shell_class=Shell,
        shell_config=shell_config,
        env_class=XOREnv,
        env_configs=[{}] * N_ENVS,
        traj_length=TRAJ_LENGTH,
        seed=SEED,
        batch_size=N_ENVS,
        exp_sender_handle=DummyReplay(),
        n_unrolls=1000,
    )

  def testInit(self):
    self._get_actor()


if __name__ == '__main__':
  tf.test.main()
