import argparse
import pdb

import numpy as np
import tensorflow as tf
from absl import flags, logging
from absl.testing import absltest
from liaison.agents import StepOutput
from liaison.agents.mlp import Agent as MLPAgent
from liaison.specs.specs import BoundedArraySpec
from liaison.utils import ConfigDict
from tensorflow.contrib.framework import nest

FLAGS = flags.FLAGS

B = 8
T = 9
N_NODES = 3
DIM = 7
EMBED_DIM = 7


class VtraceAgentTest(absltest.TestCase):

  def _get_model_config(self):
    config = ConfigDict()
    config.class_path = "liaison.agents.models.transformer"
    return config

  def _get_agent_instance(self):
    action_spec = BoundedArraySpec((10, 20),
                                   np.int32,
                                   0,
                                   N_NODES - 1,
                                   name='test_spec')

    config = ConfigDict()
    config.model = self._get_model_config()

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

    with tf.variable_scope('gcn_rins', reuse=tf.AUTO_REUSE):
      return MLPAgent(action_spec=action_spec, name='test', seed=42, **config)

  def session(self):
    return tf.Session()

  def testStep(self):
    agent = self._get_agent_instance()
    bs_ph = tf.placeholder_with_default(B, ())

    init_state = agent.initial_state(bs=bs_ph)

    step_type = np.zeros((B, ), dtype=np.int32)
    reward = np.zeros((B, ), dtype=np.float32)

    var_nodes = np.zeros((B, N_NODES, DIM), dtype=np.float32)
    var_embeddings = np.zeros((B, N_NODES, EMBED_DIM), dtype=np.int32)
    mask = np.ones((B, N_NODES), dtype=np.int32)

    obs = dict(mask=mask, var_nodes=var_nodes, var_embeddings=var_embeddings)

    step_type, reward, obs, prev_state = agent.step_preprocess(
        step_type, reward, obs, init_state)

    def f(np_arr):
      return tf.constant(np_arr)

    with tf.variable_scope('step', reuse=tf.AUTO_REUSE):
      step_output = agent.step(nest.map_structure(f, step_type),
                               nest.map_structure(f, reward),
                               nest.map_structure(f, obs), prev_state)

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

    step_type = np.zeros((T + 1, B), dtype=np.int32)
    reward = np.zeros((T + 1, B), dtype=np.float32)
    discount = np.zeros((T + 1, B), dtype=np.float32)

    var_nodes = np.zeros((T + 1, B, N_NODES, DIM), dtype=np.float32)
    var_embeddings = np.zeros((T + 1, B, N_NODES, EMBED_DIM), dtype=np.int32)
    mask = np.ones((T + 1, B, N_NODES), dtype=np.int32)

    obs = dict(mask=mask, var_nodes=var_nodes, var_embeddings=var_embeddings)

    step_output = StepOutput(action=np.zeros((T, B), dtype=np.int32),
                             logits=np.zeros((T, B, N_NODES),
                                             dtype=np.float32),
                             next_state=np.zeros_like(
                                 np.vstack([init_state_val] * T)))

    step_output, _, step_type, reward, obs, discount = agent.update_preprocess(
        step_output, None, step_type, reward, obs, discount)

    def f(np_arr):
      return tf.constant(np_arr)

    with tf.variable_scope('update', reuse=tf.AUTO_REUSE):
      agent.build_update_ops(
          nest.map_structure(f, step_output),
          tf.zeros_like(np.vstack([init_state_val] * (T + 1))),
          nest.map_structure(f, step_type), nest.map_structure(f, reward),
          nest.map_structure(f, obs), nest.map_structure(f, discount))

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for _ in range(3):
      agent.update(sess, {}, {})
      print('.', end='')
    print('')
    print('Done!')


if __name__ == '__main__':
  absltest.main()
