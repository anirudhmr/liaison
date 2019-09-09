"""TODO(arc): doc_string."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf
from agents import URDiscreteAgent
from specs import BoundedArraySpec, ArraySpec
from distributed import Learner
import numpy as np

SEED = 42
BATCH_SIZE = 8
TRAJ_LENGTH = 10


class LearnerTest(tf.test.TestCase):

  def _get_spec_handle(self):

    class DummySpec:

      @staticmethod
      def get_traj_spec(bs, traj_length):
        return dict(
            step_type=BoundedArraySpec(dtype=np.int8,
                                       shape=(TRAJ_LENGTH + 1, BATCH_SIZE),
                                       minimum=0,
                                       maximum=2,
                                       name='traj_step_type_spec'),
            reward=ArraySpec(dtype=np.float32,
                             shape=(None, None),
                             name='traj_reward_spec'),
            discount=ArraySpec(dtype=np.float32,
                               shape=(None, None),
                               name='traj_discount_spec'),
            observation=ArraySpec(dtype=np.float32,
                                  shape=(None, None),
                                  name='traj_obs_spec'),
            # simplify the specs by omitting logits, next_state here.
            step_output=dict(action=ArraySpec(
                dtype=np.int32, shape=(None, None), name='traj_action_spec')))

      @staticmethod
      def get_action_spec(*args, **kwargs):
        return BoundedArraySpec((10, 20), np.int32, 0, 100, name='action_spec')

    return DummySpec

  def _ps_publish_handle(self):

    class DummyPublishHandle:

      @staticmethod
      def publish(*args, **kwargs):
        return

    return DummyPublishHandle

  def _ps_client_handle(self):

    class DummyPSClientHandle:

      @staticmethod
      def fetch_info():
        return None

    return DummyPSClientHandle

  def _replay_handle(self):

    class DummyReplayHandle:

      @staticmethod
      def get():
        return [
            dict(step_type=np.zeros((TRAJ_LENGTH + 1, BATCH_SIZE),
                                    dtype=np.int8),
                 reward=np.zeros((TRAJ_LENGTH + 1, BATCH_SIZE),
                                 dtype=np.float32),
                 discount=np.zeros((TRAJ_LENGTH + 1, BATCH_SIZE),
                                   dtype=np.float32),
                 observation=np.zeros((TRAJ_LENGTH + 1, BATCH_SIZE),
                                      dtype=np.float32),
                 step_output=dict(
                     action=np.zeros((TRAJ_LENGTH,
                                      BATCH_SIZE), dtype=np.int32)))
        ]

    return DummyReplayHandle

  def _get_learner(self):
    return Learner(
        session_config={},
        agent_class=URDiscreteAgent,
        agent_config=dict(seed=SEED),
        spec_handle=self._get_spec_handle(),
        ps_publish_handle=self._ps_publish_handle(),
        ps_client_handle=self._ps_client_handle(),
        replay_handle=self._replay_handle(),
        batch_size=BATCH_SIZE,
        traj_length=TRAJ_LENGTH,
        publish_every=1,
        n_train_steps=1000,
    )

  def testLearner(self):
    learner = self._get_learner()
    learner.train()


if __name__ == '__main__':
  tf.test.main()
