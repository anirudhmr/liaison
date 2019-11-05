"""Start IRS Server."""

from __future__ import absolute_import, division, print_function

import json
import logging
import os
import shutil
import time
from multiprocessing import Process

import liaison.utils as U
from absl import logging
from caraml.zmq import ZmqProxyThread, ZmqServer
from liaison.utils import ConfigDict
from pathlib import Path
"""
  Request format:
    (request_type -> str, args -> List, kwargs -> Dict)

"""


class Worker(Process):

  def __init__(self, serving_host, serving_port, checkpoint_folder,
               profile_folder, **kwargs):
    Process.__init__(self)
    self.config = ConfigDict(**kwargs)
    self.checkpoint_folder = checkpoint_folder
    self.profile_folder = profile_folder
    self.serving_host = serving_host
    self.serving_port = serving_port

    # Attributes
    self._server = None

  def run(self):
    self._server = ZmqServer(host=self.serving_host,
                             port=self.serving_port,
                             serializer=U.serialize,
                             deserializer=U.deserialize,
                             bind=False)
    self._server.start_loop(handler=self._handle_request, blocking=True)

  def _handle_request(self, req):
    req_fn, args, kwargs = req
    assert isinstance(req_fn, str)
    try:
      fn = getattr(self, req_fn)
      return fn(*args, **kwargs)
    except AttributeError:
      logging.error('Unknown request func name received: %s', req_fn)

  def _read_checkpoint_info(self):
    with open(os.path.join(self.checkpoint_folder, 'info.txt'), 'r') as f:
      ckpts = []
      for line in f.readlines():
        j = json.loads(line)
        ckpts.append([j['time'], j['dst_dir_name']])

    ckpts.sort()
    return ckpts

  def _write_checkpoint_info(self, ckpts):
    with open(os.path.join(self.checkpoint_folder, 'info.txt'), 'w') as f:
      for t, d in ckpts:
        print('{"dst_dir_name": "%s", "time":%d}' % (d, t), file=f)

  def _enforce_checkpoint_policy(self):
    """
      max_to_keep should include latest checkpoint
      keep_ckpt_every_n_hrs < 0 => disables this option
    """
    max_to_keep = self.config.max_to_keep
    assert max_to_keep >= 0
    keep_ckpt_every_n_hrs = self.config.keep_ckpt_every_n_hrs
    ckpts = self._read_checkpoint_info()

    if keep_ckpt_every_n_hrs >= 0:
      prev_keep_t = ckpts[0][0]
      to_delete = []
      for i, (t, d) in enumerate(ckpts[1:]):
        if t - prev_keep_t < 3600 * keep_ckpt_every_n_hrs:
          to_delete.append([t, d])
        else:
          prev_keep_t = t
    else:
      to_delete = list(ckpts)

    if max_to_keep:
      for ckpt in ckpts[-max_to_keep:]:
        if ckpt in to_delete:
          to_delete.remove(ckpt)

    self._write_checkpoint_info(
        [ckpt for ckpt in ckpts if ckpt not in to_delete])

    for _, d in to_delete:
      U.f_remove(os.path.join(self.checkpoint_folder, d))

  def _stream_to_file(self, offset, data, fname, done):
    """fname should be with full path."""
    fname = fname.rstrip('/')

    Path(fname + '.part').touch()
    with open(fname + '.part', 'r+b') as f:
      f.seek(offset, os.SEEK_SET)
      f.write(data)

    if done:
      shutil.move(fname + '.part', fname)

  # ================== PUBLIC REMOTE API ==================
  def register_commands(self, **cmds):
    U.f_mkdir(self.config.cmd_folder)
    U.pretty_dump(cmds, os.path.join(self.config.cmd_folder, 'cmds.txt'))

  def register_metagraph(self, offset, data, _, fname, done):
    """TODO: If multi threaded client, add filelock support here."""
    U.f_mkdir(self.checkpoint_folder)
    self._stream_to_file(offset, data,
                         os.path.join(self.checkpoint_folder, fname), done)

  def register_checkpoint(self, offset, data, dst_dir_name, fname, done):
    """TODO: If multi threaded client, add filelock support here."""
    U.f_mkdir(os.path.join(self.checkpoint_folder, dst_dir_name))
    self._stream_to_file(
        offset, data, os.path.join(self.checkpoint_folder, dst_dir_name,
                                   fname), done)

    if done:
      with open(os.path.join(self.checkpoint_folder, 'info.txt'), 'a') as f:
        print('{"dst_dir_name": "%s", "time":%d}' %
              (dst_dir_name, int(time.time())),
              file=f)
      logging.info("Received new checkpoint which is saved at %s/%s/%s",
                   self.checkpoint_folder, dst_dir_name, fname)
      self.enforce_checkpoint_policy()

  def register_profile(self, offset, data, dst_dir_name, fname, done):
    """TODO: If multi threaded client, add filelock support here."""
    del dst_dir_name  # unused
    U.f_mkdir(self.profile_folder)
    self._stream_to_file(offset, data, os.path.join(self.profile_folder,
                                                    fname), done)

    if done:
      logging.info("Received new profile which is saved at %s/%s",
                   self.checkpoint_folder, fname)

  def enforce_checkpoint_policy(self):
    # Remove duplicates from checkpoint info.txt first
    ckpts = self._read_checkpoint_info()
    ckpts = sorted([[sec, first] for first, sec in ckpts])
    to_remove = []
    for i, (d, t) in enumerate(ckpts):
      if i > 0:
        if d == ckpts[i - 1][0]:
          to_remove.append(i - 1)

    self._write_checkpoint_info([[t, d] for i, (d, t) in enumerate(ckpts)
                                 if i not in to_remove])

    self._enforce_checkpoint_policy()
