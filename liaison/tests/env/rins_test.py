import pdb
from threading import Lock

import numpy as np

import tensorflow as tf
from absl import logging
from absl.testing import absltest
from liaison.env.rins import Env
from liaison.utils import ConfigDict

B = 8
T = 9
N_ACTIONS = 2
lock = Lock()


class RinsEnvTest(absltest.TestCase):

  def _get_env(self):
    return Env(0, seed=42, graph_seed=42, make_obs_for_mlp=False)

  def _print_done(self):
    with lock:
      for _ in range(10):
        print('*', end='')
      print('\nTest done')
      for _ in range(10):
        print('*', end='')
      print('\n')

  def testStep(self):
    env = self._get_env()
    ts = env.reset()
    for i in range(500):
      assert ts.reward == 0 or ts.reward <= -1
      obs = ts.observation
      mask = obs['node_mask']
      assert np.sum(mask) > 0
      act = np.random.choice(range(len(mask)), p=mask / np.sum(mask))
      ts = env.step(act)
      print(obs['log_values']['best_ep_return'],
            obs['log_values']['final_ep_return'])

    self._print_done()


if __name__ == '__main__':
  absltest.main()
