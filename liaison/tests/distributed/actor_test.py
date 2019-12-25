"""Test file for ur discrete."""

import os
import time

import liaison.utils as U
import numpy as np
import tensorflow as tf
from absl.testing import absltest
from caraml.zmq import ZmqServer
from liaison.agents import BaseAgent, StepOutput
from liaison.distributed import Actor, ExperienceCollectorServer, Shell
from liaison.distributed.parameter_server import PSRequest, PSResponse
from liaison.env import StepType, TimeStep, XOREnv
from liaison.specs.specs import ArraySpec, BoundedArraySpec

B = 8
_LOCALHOST = 'localhost'
PS_FRONTEND_PORT = '6000'
COLLECTOR_FRONTEND_PORT = '6001'
SYMPH_SPEC_PORT = '6002'
TRAJ_LENGTH = 10
N_ENVS = 2
SEED = 42


class DummyAgent(BaseAgent):

  def __init__(self, name, **kwargs):
    del kwargs
    self._name = name

    class DummyModel:

      def __init__(self):
        pass

    self._model = DummyModel()

  def _dummy_state(self, batch_size):
    return tf.fill(tf.expand_dims(batch_size, 0), 0)

  def initial_state(self, batch_size):
    return self._dummy_state(batch_size)

  def step(self, step_type, reward, obs, prev_state):
    """Pick a random discrete action from action_spec."""
    with tf.variable_scope(self._name):
      with tf.name_scope('ur_step'):
        batch_size = tf.shape(step_type)[0]
        action = tf.fill((batch_size, ), 0)
        logits = tf.fill(tf.expand_dims(batch_size, 0), 0)
        return StepOutput(action, logits, self._dummy_state(batch_size))


class DummyPS:

  def __init__(self):

    self.param_info = {
        'time': time.time(),
        'iteration': 0,
        'variable_list': [],
        'hash': U.pyobj_hash({}),
    }

    self._server = ZmqServer(
        host=_LOCALHOST,
        port=PS_FRONTEND_PORT,
        serializer=U.serialize,
        deserializer=U.deserialize,
        bind=True,
    )
    self._server_thread = self._server.start_loop(
        handler=self._handle_agent_request, blocking=False)

  def _handle_agent_request(self, request):
    """Reply to agents' request for parameters."""

    request = PSRequest(**request)

    if request.type == 'info':
      return PSResponse(type='info')._asdict()

    elif request.type == 'parameter':
      if request.hash is not None:
        if request.hash == self.param_info['hash']:  # param not changed
          return PSResponse(type='no_change', info=self.param_info)._asdict()

      return PSResponse(type='parameters', info=self.param_info,
                        parameters={})._asdict()
    else:
      raise ValueError('invalid request type received: %s' % (request.type))


class ActorTest(absltest.TestCase):

  def _setup_env(self):
    os.environ.update(
        dict(SYMPH_PS_FRONTEND_HOST=_LOCALHOST,
             SYMPH_PS_FRONTEND_PORT=PS_FRONTEND_PORT,
             SYMPH_COLLECTOR_FRONTEND_HOST=_LOCALHOST,
             SYMPH_COLLECTOR_FRONTEND_PORT=COLLECTOR_FRONTEND_PORT,
             SYMPH_SPEC_PORT=SYMPH_SPEC_PORT))

  def _setup_exp_collector(self):
    exp_server = ExperienceCollectorServer(host=_LOCALHOST,
                                           port=COLLECTOR_FRONTEND_PORT,
                                           exp_handler=lambda k: None,
                                           load_balanced=False)
    exp_server.start()
    return exp_server

  def _get_actor(self):

    self._setup_env()
    exp_server = self._setup_exp_collector()
    ps_server = DummyPS()

    shell_config = dict(
        agent_class=DummyAgent,
        agent_config={},
        sync_period=5,
        ps_client_timeout=2,
        ps_client_not_ready_sleep=2,
    )
    return Actor(actor_id=0,
                 shell_class=Shell,
                 shell_config=shell_config,
                 env_class=XOREnv,
                 env_configs=[{}] * N_ENVS,
                 traj_length=TRAJ_LENGTH,
                 seed=SEED,
                 batch_size=N_ENVS,
                 n_unrolls=1000,
                 use_full_episode_traj=True,
                 discount_factor=1.0)

  def testInit(self):
    self._get_actor()


if __name__ == '__main__':
  absltest.main()
