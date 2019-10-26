"""Test file for ur discrete."""

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from absl import logging
from liaison.agents import URDiscreteAgent
from liaison.specs.specs import BoundedArraySpec

B = 8


class URDiscreteAgentTest(tf.test.TestCase):

  def _get_agent_instance(self):
    action_spec = BoundedArraySpec((10, 20),
                                   np.int32,
                                   0,
                                   100,
                                   name='test_spec')
    return URDiscreteAgent(action_spec=action_spec, name='test', seed=42)

  def testInit(self):
    self._get_agent_instance()

  def testInitialState(self):
    agent = self._get_agent_instance()
    bs_ph = tf.placeholder_with_default(B, ())
    init_state = agent.initial_state(batch_size=bs_ph)
    with self.session(use_gpu=False) as sess:
      sess.run(init_state)

  def testStep(self):
    agent = self._get_agent_instance()
    bs_ph = tf.placeholder_with_default(B, ())

    init_state = agent.initial_state(batch_size=bs_ph)

    step_type = np.zeros((B, ), dtype=np.int32)
    reward = np.zeros((B, ), dtype=np.float32)
    obs = None
    prev_state = init_state  # hack that works for now!

    step_output = agent.step(step_type, reward, obs, prev_state)

    with self.session(use_gpu=False) as sess:
      for _ in range(100):
        sess.run(step_output)


if __name__ == '__main__':
  tf.test.main()
