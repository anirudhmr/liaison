"""Test file for ur discrete."""

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from agents import BaseAgent, StepOutput
from distributed import Shell
from env import StepType, TimeStep
from specs.specs import ArraySpec, BoundedArraySpec

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
        action = tf.fill((1, ), 0)
        logits = tf.fill(tf.expand_dims(batch_size, 0), 0)
        return StepOutput(action, logits, self._dummy_state(batch_size))


class DummyPS(object):
  count = 0

  def pull(self, var_names):
    DummyPS.count += 1
    return {var_name: 0 for var_name in var_names}


class ShellTest(tf.test.TestCase):

  def _get_shell(self):
    action_spec = BoundedArraySpec((10, 20),
                                   np.int32,
                                   0,
                                   100,
                                   name='test_spec')
    obs_spec = dict(
        state=ArraySpec(shape=(B, 10), dtype=np.float32, name='state_spec'))
    return Shell(
        action_spec,
        obs_spec,
        DummyAgent,
        {},
        DummyPS(),
        batch_size=B,
        sync_period=1,
        use_gpu=False,
    )

  def testStep(self):
    shell = self._get_shell()
    for _ in range(1000):
      shell.step(
          np.zeros((B, ), np.int32) + StepType.FIRST,
          np.zeros((B, ), np.float32) - 0.5,
          dict(state=np.zeros((B, 10), np.float32)),
      )
    # Be careful about other test functions changing the static variable
    self.assertEqual(DummyPS.count, 1000 + 1)


if __name__ == '__main__':
  tf.test.main()
