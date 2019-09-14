"""Test file for ur discrete."""

from __future__ import absolute_import, division, print_function

import os
import time

import liaison.utils as U
import numpy as np
import tensorflow as tf
from caraml.zmq import ZmqServer
from liaison.agents import BaseAgent, StepOutput
from liaison.distributed import Shell
from liaison.distributed.parameter_server import PSRequest, PSResponse
from absl.testing import absltest

from liaison.env import StepType, TimeStep
from liaison.specs.specs import ArraySpec

B = 8
_LOCALHOST = 'localhost'
PS_FRONTEND_PORT = '6000'


class DummyAgent(BaseAgent):

  def __init__(self, name, **kwargs):
    del kwargs
    self._name = name

  def _dummy_state(self, batch_size):
    return tf.fill(tf.expand_dims(batch_size, 0), 0)

  def initial_state(self, batch_size):
    return self._dummy_state(batch_size)

  def step(self, step_type, reward, obs, prev_state):
    """Pick a random discrete action from action_spec."""
    with tf.variable_scope(self._name):
      with tf.name_scope('ur_step'):
        batch_size = tf.shape(step_type)[0]
        action = tf.fill((B, ), 0)
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


class ShellTest(absltest.TestCase):

  def _setup_env(self):
    os.environ.update(
        dict(SYMPH_PS_FRONTEND_HOST=_LOCALHOST,
             SYMPH_PS_FRONTEND_PORT=PS_FRONTEND_PORT))

  def _get_shell(self):
    action_spec = ArraySpec((10, 20), np.int32, name='test_spec')
    obs_spec = dict(
        state=ArraySpec(shape=(B, 10), dtype=np.float32, name='state_spec'))
    return Shell(
        action_spec=action_spec,
        obs_spec=obs_spec,
        agent_class=DummyAgent,
        agent_config={},
        batch_size=B,
        sync_period=1,
        use_gpu=False,
        seed=42,
        ps_client_timeout=2,
        ps_client_not_ready_sleep=2,
    )

  def testStep(self):
    self._setup_env()
    ps_server = DummyPS()
    shell = self._get_shell()
    for _ in range(100):
      shell.step(
          np.zeros((B, ), np.int32) + StepType.FIRST,
          np.zeros((B, ), np.float32) - 0.5,
          dict(state=np.zeros((B, 10), np.float32)),
      )

    print('Done')


if __name__ == '__main__':
  absltest.main()
