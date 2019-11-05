import argparse
import os

import numpy as np
import tensorflow as tf
from absl import flags, logging
from absl.testing import absltest
from liaison.agents import StepOutput
from liaison.agents.gcn import Agent as GCNAgent
from liaison.specs.specs import BoundedArraySpec
from liaison.utils import ConfigDict
from tensorflow.contrib.framework import nest
from tensorflow.python.client import timeline
import networkx as nx

# use this to get large graph.
from liaison.env.utils.shortest_path import generate_networkx_graph

FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'gcn', 'Options: mlp, gcn')
flags.DEFINE_boolean('enable_gpu', True, 'Enables gpu for tf')
flags.DEFINE_integer('B', 8, 'batch_size')
flags.DEFINE_integer('T', 9, 'batch_size')

GRAPH_FEATURES = {
    # globals dimension [n_graphs, global_dim]
    "globals":
    np.array([1.1, 1.2, 1.3], dtype=np.float32),
    "nodes":
    np.array([[10.1, 10.2], [20.1, 20.2], [30.1, 30.2]], dtype=np.float32),
    "edges":
    np.array([[101., 102., 103., 104.], [201., 202., 203., 204.]],
             dtype=np.float32),
    "senders":
    np.array([0, 1], dtype=np.int32),
    "receivers":
    np.array([1, 2], dtype=np.int32),
    "n_node":
    np.array(3, dtype=np.int32),
    "n_edge":
    np.array(2, dtype=np.int32)
}
N_NODES = 3
SEED = 42

# set after parsing flags
B = None
T = None


class VtraceAgentTest(absltest.TestCase):

  def _setup(self):
    global B, T
    B = FLAGS.B
    T = FLAGS.T

  def _large_graph(self):
    # generate graph with 32 nodes.
    nx_graph, path = generate_networkx_graph(SEED, [32, 33])
    src_node, target_node = path[0], path[-1]
    nx_graph = nx_graph.to_directed()
    # Create graph features from the networkx graph
    # make sure to set all the static fields in the created features.
    # also initialize the dynamic fields to the right values.
    nodes = np.zeros((len(nx_graph), 6), dtype=np.float32)

    edges = np.zeros([nx_graph.number_of_edges(), 2], dtype=np.float32)
    weights = nx.get_edge_attributes(nx_graph, 'distance')
    for i, edge in enumerate(nx_graph.edges()):
      edges[i][0] = weights[edge]

    senders, receivers = zip(*nx_graph.edges())
    graph = dict(nodes=nodes,
                 edges=edges,
                 globals=np.zeros(1, dtype=np.float32),
                 senders=np.array(senders, dtype=np.int32),
                 receivers=np.array(receivers, dtype=np.int32),
                 n_node=np.array(len(nodes), dtype=np.int32),
                 n_edge=np.array(len(edges), dtype=np.int32))
    return graph

  def _get_model_config(self):
    config = ConfigDict()
    if FLAGS.model == 'mlp':
      config.class_path = "liaison.agents.models.mlp"
      config.hidden_layer_sizes = [32, 32]
    elif FLAGS.model == 'gcn':
      config.class_path = "liaison.agents.models.gcn"
    else:
      raise Exception('Unknown model %s' % FLAGS.model)
    return config

  def _get_agent_instance(self):
    action_spec = BoundedArraySpec((10, 20),
                                   np.int32,
                                   0,
                                   100,
                                   name='test_spec')

    config = ConfigDict()
    config.model = self._get_model_config()

    config.lr_init = 1e-3
    config.lr_min = 1e-4
    config.lr_start_dec_step = 1000
    config.lr_dec_steps = 1000
    config.lr_dec_val = .1
    config.lr_dec_approach = 'linear'

    config.ent_dec_init = 1
    config.ent_dec_min = 0
    config.ent_dec_steps = 1000
    config.ent_start_dec_step = 1000
    config.ent_dec_val = .1
    config.ent_dec_approach = 'linear'

    config.grad_clip = 1.0
    config.discount_factor = 0.99
    config.clip_rho_threshold = 1.0
    config.clip_pg_rho_threshold = 1.0

    config.loss = ConfigDict()
    config.loss.vf_loss_coeff = 1.0

    with tf.variable_scope('gcn', reuse=tf.AUTO_REUSE):
      return GCNAgent(action_spec=action_spec, name='test', seed=42, **config)

  def session(self):
    if FLAGS.enable_gpu:
      return tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))
    return tf.Session()

  def _get_graph_features_update(self, large_graph=True):
    # get timestep stacked and batched graph features
    def f(*l):
      return np.stack(l, axis=0)

    if large_graph:
      graph_features = self._large_graph()
    else:
      graph_features = GRAPH_FEATURES
    graph_features = nest.map_structure(f, *[graph_features] * B)
    return nest.map_structure(f, *[graph_features] * (T + 1))

  def testUpdate(self):
    self._setup()
    agent = self._get_agent_instance()
    bs_ph = tf.placeholder_with_default(B, ())
    sess = self.session()

    init_state = agent.initial_state(bs=bs_ph)
    init_state_val = sess.run(init_state)

    step_type = np.zeros((T + 1, B), dtype=np.int32)
    reward = np.zeros((T + 1, B), dtype=np.float32)
    discount = np.zeros((T + 1, B), dtype=np.float32)
    obs = dict(features=np.zeros((T + 1, B, N_NODES), dtype=np.float32),
               graph_features=self._get_graph_features_update(),
               node_mask=np.ones(((T + 1), B, N_NODES), dtype=np.int32))

    step_output = StepOutput(action=np.zeros((T, B), dtype=np.int32),
                             logits=np.zeros((T, B, N_NODES),
                                             dtype=np.float32),
                             next_state=np.zeros_like(
                                 np.vstack([init_state_val] * T)))

    step_output, _, step_type, reward, obs, discount = agent.update_preprocess(
        step_output, None, step_type, reward, obs, discount)

    feed_dict = {}

    def f(np_arr):
      ph = tf.placeholder(shape=np_arr.shape, dtype=np_arr.dtype)
      feed_dict[ph] = np_arr
      return ph

    with tf.variable_scope('update', reuse=tf.AUTO_REUSE):
      agent.build_update_ops(
          nest.map_structure(f, step_output),
          tf.zeros_like(np.vstack([init_state_val] * (T + 1))),
          nest.map_structure(f, step_type), nest.map_structure(f, reward),
          nest.map_structure(f, obs), nest.map_structure(f, discount))

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    N_ITERS = 50
    for i in range(N_ITERS):
      profile_kwargs = {}
      if i == N_ITERS - 1:
        run_metadata = tf.RunMetadata()
        profile_kwargs = dict(
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            run_metadata=run_metadata)

      agent.update(sess, feed_dict, profile_kwargs)
      print('.', end='')

    print('')

    # save the final timeline
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    export_path = '/tmp/'
    with open(os.path.join(export_path, 'timeline.json'), 'w') as f:
      f.write(ctf)
    print('Done!')


if __name__ == '__main__':
  absltest.main()
