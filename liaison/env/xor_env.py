import numpy as np
from env import Env
from env.environment import restart, termination
from specs import BoundedArraySpec


class XOREnv(Env):
  """Single Step environment with reward equal to 1
  for correct XOR value and 0 otherwise."""

  def __init__(self, id, seed, **env_config):
    self.id = id
    self.seed = seed
    self.set_seed(seed)
    self._features = None
    self._reset_next_step = True

  def set_seed(self, seed):
    np.random.seed(seed + self.id)

  def reset(self):
    self._reset_next_step = False
    self._features = np.float32(np.random.randint(0, 2, size=(2, )))
    return restart(self._observation())

  def _observation(self):
    obs = dict(features=np.float32(self._features))
    return obs

  def step(self, action):
    if self._reset_next_step:
      return self.reset()

    self._reset_next_step = True

    action = int(action)
    ans = np.logical_xor(self._features[0], self._features[1])
    rew = np.float32(ans == action)
    return termination(rew, self._observation())

  def observation_spec(self):
    features_spec = BoundedArraySpec((2, ),
                                     np.float32,
                                     0,
                                     1,
                                     name='features_spec')
    return dict(features=features_spec)

  def action_spec(self):
    return BoundedArraySpec((), np.int32, 0, 1, name='action_spec')
