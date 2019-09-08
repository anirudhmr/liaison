"""Test file for ur discrete."""

from __future__ import absolute_import, division, print_function

import os
import time

import numpy as np
import tensorflow as tf
from absl import logging
from agents import BaseAgent, StepOutput
from distributed import (ParameterClient, ParameterPublisher,
                         ShardedParameterServer, Shell)
from env import StepType, TimeStep
from specs.specs import ArraySpec, BoundedArraySpec

SYMPH_PARAMETER_PUBLISH_HOST = "localhost"
SYMPH_PARAMETER_PUBLISH_PORT = "6000"

N_SHARDS = 8
SYMPH_PS_FRONTEND_HOST = "localhost"
SYMPH_PS_FRONTEND_PORT = "6001"
SYMPH_PS_BACKEND_PORT = "6002"


class ParameterServerTest(tf.test.TestCase):

  def _get_ps_publisher(self):

    return ParameterPublisher(port=SYMPH_PARAMETER_PUBLISH_PORT)

  def _get_ps_client(self):
    return ParameterClient(port=SYMPH_PS_FRONTEND_PORT,
                           host=SYMPH_PS_FRONTEND_HOST,
                           timeout=0.1)

  def _get_ps(self):
    return ShardedParameterServer(shards=N_SHARDS, supress_output=True)

  def _setup_env(self):
    os.environ.update(
        dict(SYMPH_PS_FRONTEND_PORT=SYMPH_PS_FRONTEND_PORT,
             SYMPH_PS_BACKEND_PORT=SYMPH_PS_BACKEND_PORT,
             SYMPH_PARAMETER_PUBLISH_PORT=SYMPH_PARAMETER_PUBLISH_PORT,
             SYMPH_PARAMETER_PUBLISH_HOST=SYMPH_PARAMETER_PUBLISH_HOST))

  def testSetup(self):
    self._setup_env()
    logging.info('Creating ps publisher')
    pub = self._get_ps_publisher()
    logging.info('Creating parameter server')
    ps = self._get_ps()
    logging.info('Launching parameter server')
    ps.launch()
    logging.info('Creating ps client')
    cli = self._get_ps_client()
    # make sure that subscriber is ready to listen to publishings.
    time.sleep(2)

    for i in range(1000):
      var_dict = dict(x=np.array(i, dtype=np.int32),
                      y=np.array(i, dtype=np.int32))
      pub.publish(i, var_dict)
      vars_to_fetch = ['x']
      while True:
        param, info = cli.fetch_parameter_with_info(vars_to_fetch)
        if info is None:
          logging.info('Received None as info')
        elif info['iteration'] != i:
          logging.info('iteration received type: {type}, val: {val}'.format(
              type=type(info['iteration']), val=info['iteration']))
          # wait for the latest copy to arrive.
        else:
          break

      self.assertNotEqual(info, None)
      self.assertEqual(info['iteration'], i)
      self.assertListEqual(info['variable_list'], list(var_dict.keys()))
      self.assertListEqual(sorted(param.keys()), sorted(vars_to_fetch))
      self.assertEqual(
          param, {var_name: var_dict[var_name]
                  for var_name in vars_to_fetch})
      logging.info('Processed iteration %d successfully', i)

    ps.quit()
    time.sleep(1)


if __name__ == '__main__':
  tf.test.main()
