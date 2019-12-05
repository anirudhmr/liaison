import copy
import pdb
from threading import Lock

import numpy as np

import tensorflow as tf
import tree as nest
from absl import logging
from absl.testing import absltest
from liaison.agents.mlp import Agent as MLPAgent
from liaison.env.batch import ParallelBatchedEnv, SerialBatchedEnv
from liaison.env.rins import Env
from liaison.specs.specs import BoundedArraySpec
from liaison.utils import ConfigDict

B = 8
T = 9
N_ACTIONS = 2
SEED = 42
lock = Lock()


def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()


class RinsEnvTest(absltest.TestCase):

  def _get_env(self):
    env = SerialBatchedEnv(B, Env, [
        dict(graph_seed=42,
             make_obs_for_mlp=True,
             make_obs_for_self_attention=False,
             mlp_embed_constraints=False,
             k=40,
             steps_per_episode=180,
             dataset='milp-facilities-10')
    ] * B, SEED)
    return env

  def _print_done(self):
    with lock:
      for _ in range(10):
        print('*', end='')
      print('\nTest done')
      for _ in range(10):
        print('*', end='')
      print('\n')

  def _get_agent(self, action_spec):
    config = ConfigDict()
    config.model = ConfigDict()
    config.model.class_path = "liaison.agents.models.mlp"
    config.model.hidden_layer_sizes = [64, 64]

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
    with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
      return MLPAgent(action_spec=action_spec, name='test', seed=42, **config)

  def _mk_phs(self, obs_spec, initial_state_dummy_spec):

    def mk_ph(spec):
      return tf.placeholder(dtype=spec.dtype,
                            shape=spec.shape,
                            name='shell_' + spec.name + '_ph')

    self._step_type_ph = tf.placeholder(dtype=tf.int8,
                                        shape=(None, ),
                                        name='shell_step_type_ph')
    self._reward_ph = tf.placeholder(dtype=tf.float32,
                                     shape=(None, ),
                                     name='shell_reward_ph')
    self._obs_ph = nest.map_structure(mk_ph, obs_spec)
    self._next_state_ph = tf.placeholder(dtype=initial_state_dummy_spec.dtype,
                                         shape=initial_state_dummy_spec.shape,
                                         name='next_state_ph')

  def session(self):
    return tf.Session()

  def testStep(self):

    sess = tf.Session()
    env = self._get_env()
    ts = env.reset()

    agent = self._get_agent(env.action_spec())
    bs_ph = tf.placeholder_with_default(B, ())
    self._initial_state_op = agent.initial_state(bs_ph)
    dummy_initial_state = sess.run(self._initial_state_op)
    self._mk_phs(env.observation_spec(), dummy_initial_state)

    step_output_op = agent.step(self._step_type_ph, self._reward_ph,
                                copy.copy(self._obs_ph), self._next_state_ph)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    next_state = dummy_initial_state
    for _ in range(3):
      step_type = ts.step_type
      reward = ts.reward
      observation = ts.observation

      # print('..............')
      # print('Zeroing out observations')
      # observation['features'] = np.ones_like(observation['features'])
      obs_feed_dict = {
          obs_ph: obs_val
          for obs_ph, obs_val in zip(nest.flatten(self._obs_ph),
                                     nest.flatten(observation))
      }
      step_output = sess.run(step_output_op,
                             feed_dict={
                                 self._step_type_ph: step_type,
                                 self._reward_ph: reward,
                                 bs_ph: B,
                                 self._next_state_ph: next_state,
                                 **obs_feed_dict,
                             })
      next_state = step_output.next_state
      logits = step_output.logits
      print(logits)
      print(step_output.action)
      ts = env.step(step_output.action)

      pdb.set_trace()
      print('.', end='')
    print('')
    print('Done!')


if __name__ == '__main__':
  absltest.main()
