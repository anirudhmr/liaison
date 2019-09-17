"""Server to host the trajectory specs needed by learner."""
from __future__ import absolute_import, division, print_function

from threading import Thread

import liaison.utils as U
from absl import logging
from caraml.zmq import ZmqServer

from .trajectory import Trajectory


class SpecServer(Thread):

  def __init__(self, port, traj_spec, action_spec):
    self._traj_spec = traj_spec
    self._action_spec = action_spec
    self.port = port
    super(SpecServer, self).__init__()

  def run(self):
    self._server = ZmqServer(
        host='*',
        port=self.port,
        serializer=U.pickle_serialize,
        deserializer=U.pickle_deserialize,
        bind=True,
    )

    self._server_thread = self._server.start_loop(handler=self._handle_request,
                                                  blocking=False)
    logging.info('Spec server started')

    self._server_thread.join()

  def _handle_request(self, req):
    """req -> (batch_size, traj_length)"""
    batch_size, _ = req
    traj_spec = Trajectory.format_traj_spec(self._traj_spec, *req)
    self._action_spec.set_shape((batch_size, ) + self._action_spec.shape[1:])
    return traj_spec, self._action_spec
