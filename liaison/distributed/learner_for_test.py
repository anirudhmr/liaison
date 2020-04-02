import copy
import logging
import os
import uuid
from queue import Queue
from threading import Thread

import liaison.utils as U
import tensorflow as tf
from liaison.distributed import Trajectory
from liaison.session.tracker import PeriodicTracker
from liaison.utils import ConfigDict, logging
from tensorflow.contrib.framework import nest
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import timeline


class Learner(object):

  def __init__(self,
               agent_class,
               agent_config,
               seed,
               traj_spec,
               action_spec,
               agent_scope='learner',
               use_gpu=True,
               **unused_kwargs):
    self._agent_scope = agent_scope
    self._graph = tf.Graph()
    self._traj_spec = traj_spec

    with self._graph.as_default():
      self._global_step_op = tf.train.get_or_create_global_step(self._graph)

      self._agent = agent_class(name=agent_scope,
                                action_spec=action_spec,
                                seed=seed,
                                **agent_config)

      self._mk_phs(self._traj_spec)
      traj_phs = self._traj_phs
      self._agent.build_update_ops(step_types=traj_phs['step_type'],
                                   prev_states=traj_phs['step_output']['next_state'],
                                   step_outputs=ConfigDict(traj_phs['step_output']),
                                   observations=copy.copy(traj_phs['observation']),
                                   rewards=traj_phs['reward'],
                                   discounts=traj_phs['discount'])

      config = tf.ConfigProto()
      if use_gpu:
        config.gpu_options.allow_growth = True
        # config.intra_op_parallelism_threads = 1
        # config.inter_op_parallelism_threads = 1
        self.sess = tf.Session(config=config)
      else:
        config.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.OFF
        config.device_count = {'GPU': 0}
        self.sess = tf.Session(config=config)

      self.sess.run(tf.global_variables_initializer())
      self.sess.run(tf.local_variables_initializer())

  def _mk_phs(self, traj_spec):

    def mk_ph(spec):
      return tf.placeholder(dtype=spec.dtype,
                            shape=spec.shape,
                            name='learner/' + spec.name.replace(':', '_') + '_ph')

    self._traj_phs = nest.map_structure(mk_ph, traj_spec)

  def batch_and_preprocess_trajs(self, l):
    traj = Trajectory.batch(l, self._traj_spec)
    # feed and overwrite the trajectory
    traj['step_output'], traj['step_output']['next_state'], traj['step_type'], traj[
        'reward'], traj['observation'], traj['discount'] = self._agent.update_preprocess(
            step_outputs=ConfigDict(traj['step_output']),
            prev_states=traj['step_output']['next_state'],
            step_types=traj['step_type'],
            rewards=traj['reward'],
            observations=traj['observation'],
            discounts=traj['discount'])
    return traj

  @property
  def global_step(self):
    return self.sess.run(self._global_step_op)

  def update(self, batch):
    # run update step on the sampled batch
    feed_dict = {ph: val for ph, val in zip(nest.flatten(self._traj_phs), nest.flatten(batch))}
    log_vals = self._agent.update(self.sess, feed_dict, {})
    return log_vals
