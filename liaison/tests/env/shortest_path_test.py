import numpy as np
import tensorflow as tf
from absl import logging
from absl.testing import absltest

from liaison.env.shortest_path import Env
from liaison.utils import ConfigDict

B = 8
T = 9
N_ACTIONS = 2


class ShortestPathEnvTest(absltest.TestCase):

  def _get_env(self):
    return Env(0, seed=42, make_obs_for_mlp=False)

  def testStep(self):
    env = self._get_env()
    ts = env.reset()
    for i in range(1000):
      obs = ts.observation
      mask = obs['node_mask']
      assert np.sum(mask) > 0
      act = np.random.choice(range(len(mask)), p=mask / np.sum(mask))
      ts = env.step(act)
    for _ in range(10):
      print('*', end='')
    print('\nTest done')
    for _ in range(10):
      print('*', end='')
    print('\n')


if __name__ == '__main__':
  absltest.main()
