import numpy as np
import tensorflow as tf
from absl import logging
from absl.testing import absltest

from liaison.agents import StepOutput
from liaison.agents.vtrace import Agent as VTraceAgent
from liaison.specs.specs import BoundedArraySpec
from liaison.utils import ConfigDict

B = 8
T = 9
N_ACTIONS = 2


class VtraceAgentTest(absltest.TestCase):

  def _get_agent_instance(self):
    action_spec = BoundedArraySpec((10, 20),
                                   np.int32,
                                   0,
                                   100,
                                   name='test_spec')

    config = ConfigDict()
    config.model = ConfigDict()
    config.model.class_path = "liaison.agents.models.mlp"
    config.model.hidden_layer_sizes = [32, 32]
    config.model.n_actions = N_ACTIONS

    config.lr_init = 1e-3
    config.lr_min = 1e-4
    config.lr_start_dec_step = 1000
    config.lr_dec_steps = 1000
    config.lr_dec_val = .1
    config.lr_dec_approach = 'linear'

    config.ent_dec_init = 1
    config.ent_dec_min = 0
    config.ent_dec_steps = 1000
    config.ent_start_dec_step = 1000
    config.ent_dec_val = .1
    config.ent_dec_approach = 'linear'

    config.grad_clip = 1.0
    config.discount_factor = 0.99
    config.clip_rho_threshold = 1.0
    config.clip_pg_rho_threshold = 1.0

    config.loss = ConfigDict()
    config.loss.vf_loss_coeff = 1.0

    return VTraceAgent(action_spec=action_spec, name='test', seed=42, **config)

  def session(self):
    return tf.Session()

  def testStep(self):
    agent = self._get_agent_instance()
    bs_ph = tf.placeholder_with_default(B, ())

    init_state = agent.initial_state(bs=bs_ph)

    step_type = tf.zeros((B, ), dtype=np.int32)
    reward = tf.zeros((B, ), dtype=np.float32)
    obs = dict(features=tf.zeros((B, 6), dtype=np.float32))
    prev_state = init_state  # hack that works for now!

    with tf.variable_scope('step', reuse=tf.AUTO_REUSE):
      step_output = agent.step(step_type, reward, obs, prev_state)

    sess = self.session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for _ in range(100):
      sess.run(step_output)
      print('.', end='')
    print('')
    print('Done!')

  def testUpdate(self):
    agent = self._get_agent_instance()
    bs_ph = tf.placeholder_with_default(B, ())
    sess = self.session()

    init_state = agent.initial_state(bs=bs_ph)
    init_state_val = sess.run(init_state)

    step_type = tf.zeros((T + 1, B), dtype=np.int32)
    reward = tf.zeros((T + 1, B), dtype=np.float32)
    discount = tf.zeros((T + 1, B), dtype=np.float32)
    obs = dict(features=tf.zeros((T + 1, B, 6), dtype=np.float32))

    step_output = StepOutput(action=tf.zeros((T, B), dtype=tf.int32),
                             logits=tf.zeros((T, B, N_ACTIONS),
                                             dtype=tf.float32),
                             next_state=tf.zeros_like(
                                 np.vstack([init_state_val] * T)))

    with tf.variable_scope('update', reuse=tf.AUTO_REUSE):
      agent.build_update_ops(
          step_output, tf.zeros_like(np.vstack([init_state_val] * (T + 1))),
          step_type, reward, obs, discount)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for _ in range(100):
      agent.update(sess, {})
      print('.', end='')
    print('')
    print('Done!')


if __name__ == '__main__':
  absltest.main()
