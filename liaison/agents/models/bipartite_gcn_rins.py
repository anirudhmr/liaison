"""Graphnet based model."""
import numpy as np

import graph_nets as gn
import sonnet as snt
import tensorflow.keras as K
import tree as nest
from liaison.agents.models.bipartite_gcn import BipartiteGraphConvolution
from liaison.agents.models.gcn_rins import INF
from liaison.agents.models.gcn_rins import Model as GCNRinsModel
from liaison.agents.models.gcn_rins import make_mlp
from liaison.agents.models.utils import *
from liaison.agents.utils import infer_shape
from liaison.utils import ConfigDict
from sonnet.python.ops import initializers


class Model(GCNRinsModel):

  def __init__(self,
               seed,
               activation='relu',
               n_prop_layers=4,
               edge_embed_dim=32,
               node_embed_dim=32,
               global_embed_dim=8,
               policy_torso_hidden_layer_sizes=[64, 64],
               value_torso_hidden_layer_sizes=[64, 64],
               action_spec=None,
               sum_aggregation=True,
               memory_hack=False,
               **kwargs):
    self.activation = get_activation_from_str(activation)
    self.n_prop_layers = n_prop_layers
    self.seed = seed
    self.policy = None
    self.value = None
    self.edge_embed_dim = edge_embed_dim
    self.node_embed_dim = node_embed_dim
    self.global_embed_dim = global_embed_dim
    self.config = ConfigDict(kwargs)

    with tf.variable_scope('encode'):

      def f(embed_dim, name):
        return make_mlp([embed_dim] * 2,
                        activation,
                        activate_final=True,
                        seed=seed,
                        layer_norm=False,
                        name=name)

      self._var_encode_net = f(node_embed_dim, 'var_encode_net')
      self._constraint_encode_net = f(node_embed_dim, 'constraint_encode_net')
      self._obj_encode_net = f(node_embed_dim, 'obj_encode_net')

      self._encode_net = gn.modules.BipartiteGraphIndependent(
          edge_model_fn=lambda: f(edge_embed_dim, 'edge_encode'),
          global_model_fn=lambda: f(global_embed_dim, 'global_encode'),
          left_node_model_fn=lambda: f(node_embed_dim, 'left_node_encode'),
          right_node_model_fn=lambda: f(node_embed_dim, 'right_node_encode'),
      )

    self.initializer = K.initializers.Orthogonal()
    # GRAPH CONVOLUTIONS
    # TODO: Handle the case where edge_embed_dim is different from node_embed_dim
    with tf.variable_scope('graph_conv_v_to_c'):
      self.conv_v_to_c = BipartiteGraphConvolution(self.node_embed_dim,
                                                   self.activation,
                                                   self.initializer,
                                                   right_to_left=False,
                                                   sum_aggregation=sum_aggregation,
                                                   memory_hack=memory_hack)

    with tf.variable_scope('graph_conv_c_to_v'):
      self.conv_c_to_v = BipartiteGraphConvolution(self.node_embed_dim,
                                                   self.activation,
                                                   self.initializer,
                                                   right_to_left=True,
                                                   sum_aggregation=sum_aggregation,
                                                   memory_hack=memory_hack)

    with tf.variable_scope('policy_torso'):
      self.policy_torso = snt.nets.MLP(policy_torso_hidden_layer_sizes + [1],
                                       initializers=dict(w=glorot_uniform(seed),
                                                         b=initializers.init_ops.Constant(0)),
                                       activate_final=False,
                                       activation=self.activation)

    with tf.variable_scope('value_torso'):
      # Apply this before pooling
      self.value_torso_1_left = snt.nets.MLP(
          value_torso_hidden_layer_sizes,
          initializers=dict(w=glorot_uniform(seed), b=initializers.init_ops.Constant(0)),
          activate_final=True,
          activation=self.activation,
      )
      self.value_torso_1_right = snt.nets.MLP(
          value_torso_hidden_layer_sizes,
          initializers=dict(w=glorot_uniform(seed), b=initializers.init_ops.Constant(0)),
          activate_final=True,
          activation=self.activation,
      )
      # Apply this after mean pooling of all the node features.
      self.value_torso_2 = snt.nets.MLP(
          value_torso_hidden_layer_sizes + [1],
          initializers=dict(w=glorot_uniform(seed), b=initializers.init_ops.Constant(0)),
          activate_final=False,
          activation=self.activation,
      )

  def _validate_observations(self, obs):
    for k in ['graph_features', 'node_mask']:
      if k not in obs:
        raise Exception('%s not found in observation.' % k)

  def _encode(self, graph_features):
    return self._encode_net(graph_features)

  def _convolve(self, graph_features):
    left_features = graph_features.left_nodes
    right_features = graph_features.right_nodes

    for i in range(self.n_prop_layers):
      with tf.variable_scope('prop_layer_%d' % i):
        with tf.variable_scope('v_to_c'):
          right_features = self.conv_v_to_c(
              (left_features, (graph_features.senders, graph_features.receivers),
               graph_features.edges, right_features))
        right_features = self.activation(right_features)

        with tf.variable_scope('c_to_c'):
          left_features = self.conv_c_to_v(
              (left_features, (graph_features.senders, graph_features.receivers),
               graph_features.edges, right_features))
        left_features = self.activation(left_features)

        # TODO: Add residuals?
        # TODO: Add layernorm?
        # Note edge and global modules are missing here

    graph_features = graph_features.replace(right_nodes=right_features, left_nodes=left_features)
    return graph_features

  def compute_graph_embeddings(self, obs):
    self._validate_observations(obs)
    # Run multiple rounds of graph convolutions
    ge = self._convolve(self._encode(obs['graph_features']))
    return ge

  def get_node_embeddings(self, obs, graph_embeddings):
    # Returns embeddings of the nodes in the shape (B, N_max, d)
    graph_features = graph_embeddings
    broadcasted_globals = gn.utils_tf.repeat(graph_features.globals,
                                             graph_features.n_left_nodes,
                                             axis=0)
    left_nodes = graph_features.left_nodes[:infer_shape(broadcasted_globals)[0]]
    left_nodes = tf.concat([left_nodes, broadcasted_globals], axis=-1)
    indices = gn.utils_tf.sparse_to_dense_indices(graph_features.n_left_nodes)
    left_nodes = tf.scatter_nd(indices, left_nodes,
                               infer_shape(obs['node_mask']) + [infer_shape(left_nodes)[-1]])
    return left_nodes, graph_features.n_left_nodes

  def get_logits(self, graph_features, node_mask):
    """
      graph_embeddings: Message propagated graph embeddings.
                        Use self.compute_graph_embeddings to compute and cache
                        these to use with different network heads for value, policy etc.
    """
    # broadcast globals and attach them to node features
    broadcasted_globals = gn.utils_tf.repeat(graph_features.globals,
                                             graph_features.n_left_nodes,
                                             axis=0)
    left_nodes = tf.concat([graph_features.left_nodes, broadcasted_globals], axis=-1)

    # get logits over nodes
    logits = self.policy_torso(left_nodes)
    # remove the final singleton dimension
    logits = tf.squeeze(logits, axis=-1)
    log_vals = {}
    # record norm *before* adding -INF to invalid spots
    log_vals['opt/logits_norm'] = tf.linalg.norm(logits)

    indices = gn.utils_tf.sparse_to_dense_indices(graph_features.n_left_nodes)
    logits = tf.scatter_nd(indices, logits, tf.shape(node_mask))
    logits = tf.where(tf.equal(node_mask, 1), logits, tf.fill(tf.shape(node_mask), -INF))
    return logits, log_vals

  def get_auxiliary_loss(self, graph_features: gn.graphs.GraphsTuple, obs):
    """
      Returns a prediction for each node.
      This is useful for supervised node labelling/prediction tasks.
    """
    # broadcast globals and attach them to node features
    broadcasted_globals = gn.utils_tf.repeat(graph_features.globals,
                                             graph_features.n_left_nodes,
                                             axis=0)
    left_nodes = tf.concat([graph_features.left_nodes, broadcasted_globals], axis=-1)

    # get logits over nodes
    preds = self.supervised_prediction_torso(left_nodes)
    # remove the final singleton dimension
    preds = tf.squeeze(preds, axis=-1)

    indices = gn.utils_tf.sparse_to_dense_indices(graph_features.n_left_nodes)
    opt_sol = tf.gather_nd(indices, obs['optimal_solution'], infer_shape(preds))

    auxiliary_loss = tf.reduce_mean((preds - opt_sol)**2)
    return auxiliary_loss

  def get_value(self, graph_features: gn.graphs.GraphsTuple):
    """
      graph_embeddings: Message propagated graph embeddings.
                        Use self.compute_graph_embeddings to compute and cache
                        these to use with different network heads for value, policy etc.
    """
    with tf.variable_scope('value_network'):
      left_nodes = graph_features.left_nodes
      right_nodes = graph_features.right_nodes
      num_graphs = infer_shape(graph_features.n_left_nodes)[0]

      # Apply mlp to left and right nodes.
      left_nodes = self.value_torso_1_left(left_nodes)
      right_nodes = self.value_torso_1_right(right_nodes)

      # aggregate left and right nodes seperately.
      left_indices = gn.utils_tf.repeat(tf.range(num_graphs), graph_features.n_left_nodes, axis=0)
      left_agg = tf.unsorted_segment_mean(left_nodes[:infer_shape(left_indices)[0]], left_indices,
                                          num_graphs)

      right_indices = gn.utils_tf.repeat(tf.range(num_graphs),
                                         graph_features.n_right_nodes,
                                         axis=0)
      right_agg = tf.unsorted_segment_mean(right_nodes[:infer_shape(right_indices)[0]],
                                           right_indices, num_graphs)

      value = tf.concat([left_agg, right_agg, graph_features.globals], axis=-1)
      return tf.squeeze(self.value_torso_2(value), axis=-1)

  def dummy_state(self, bs):
    return tf.fill(tf.expand_dims(bs, 0), 0)

  def get_initial_state(self, bs):
    return self.dummy_state(bs)
