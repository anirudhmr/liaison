"""Test file for ur discrete."""

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from absl import logging
from liaison.agents import URDiscreteAgent
from liaison.specs.specs import BoundedArraySpec

T = 6
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
    obs = dict()
    prev_state = init_state  # hack that works for now!

    step_output = agent.step(step_type, reward, obs, prev_state)

    with self.session(use_gpu=False) as sess:
      for _ in range(100):
        sess.run(step_output)

  def testUpdate(self):
    agent = self._get_agent_instance()
    bs_ph = tf.placeholder_with_default(B, ())

    init_state = agent.initial_state(batch_size=bs_ph)

    step_type = np.ones((T + 1, B), dtype=np.int32)
    reward = np.zeros((T + 1, B), dtype=np.float32)
    obs = dict(log_values=dict(x=np.ones((T + 1, B), dtype=np.float32)))
    prev_state = init_state  # hack that works for now!

    step_output = agent.build_update_ops(None, None, step_type, reward, obs,
                                         prev_state)

    with self.session(use_gpu=False) as sess:
      sess.run(tf.global_variables_initializer())
      for _ in range(3):
        print(agent.update(sess, {}, None))


if __name__ == '__main__':
  tf.test.main()
