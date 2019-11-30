"""Graphnet based model."""

import numpy as np

import graph_nets as gn
import sonnet as snt
from liaison.agents.models.utils import *
from sonnet.python.ops import initializers

INF = np.float32(1e9)

EDGE_BLOCK_OPT = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": False,
}

NODE_BLOCK_OPT = {
    "use_received_edges": True,
    "use_sent_edges": False,
    "use_nodes": True,
    "use_globals": True,
}

GLOBAL_BLOCK_OPT = {
    "use_edges": False,
    "use_nodes": False,
    "use_globals": True,
}


def make_mlp(layer_sizes,
             activation,
             activate_final,
             seed,
             layer_norm=False,
             **kwargs):
  mlp = snt.nets.MLP(layer_sizes,
                     initializers=dict(w=glorot_uniform(seed),
                                       b=initializers.init_ops.Constant(0)),
                     activate_final=activate_final,
                     activation=get_activation_from_str(activation),
                     **kwargs)
  if layer_norm:
    return snt.Sequential([mlp, snt.LayerNorm()])
  else:
    return mlp


class Model:

  def __init__(self,
               seed,
               activation='relu',
               n_prop_layers=8,
               edge_embed_dim=32,
               node_embed_dim=32,
               global_embed_dim=8,
               node_hidden_layer_sizes=[64, 64],
               edge_hidden_layer_sizes=[64, 64],
               policy_torso_hidden_layer_sizes=[64, 64],
               value_torso_hidden_layer_sizes=[64, 64],
               action_spec=None):
    self.activation = activation
    self.n_prop_layers = n_prop_layers
    self.seed = seed
    self.policy = None
    self.value = None

    with tf.variable_scope('encode'):
      self._var_encode_net = snt.Linear(node_embed_dim, name='var_encode_net')
      self._constraint_encode_net = snt.Linear(node_embed_dim,
                                               name='constraint_encode_net')
      self._obj_encode_net = snt.Linear(node_embed_dim, name='obj_encode_net')
      self._encode_net = gn.modules.GraphIndependent(
          edge_model_fn=lambda: snt.Linear(edge_embed_dim, name='edge_encode'),
          global_model_fn=lambda: snt.Linear(global_embed_dim,
                                             name='global_encode'),
          node_model_fn=None)

    self._graphnet_models = [None for _ in range(n_prop_layers)]
    for i in range(n_prop_layers):
      with tf.variable_scope('graphnet_model_%d' % i):
        with tf.variable_scope('edge_model'):
          edge_model = make_mlp(edge_hidden_layer_sizes + [edge_embed_dim],
                                activation,
                                activate_final=False,
                                seed=seed,
                                layer_norm=True)
        with tf.variable_scope('node_model'):
          node_model = make_mlp(node_hidden_layer_sizes + [node_embed_dim],
                                activation,
                                activate_final=False,
                                seed=seed,
                                layer_norm=True)
        # global(node(edge))
        self._graphnet_models[i] = gn.modules.GraphNetwork(
            edge_model_fn=lambda: edge_model,
            node_model_fn=lambda: node_model,
            global_model_fn=lambda: lambda x:
            x,  # Don't summarize nodes/edges to the globals.
            node_block_opt=NODE_BLOCK_OPT,
            edge_block_opt=EDGE_BLOCK_OPT,
            global_block_opt=GLOBAL_BLOCK_OPT)

    with tf.variable_scope('policy_torso'):
      self.policy_torso = snt.nets.MLP(
          policy_torso_hidden_layer_sizes + [1],
          initializers=dict(w=glorot_uniform(seed),
                            b=initializers.init_ops.Constant(0)),
          activate_final=False,
          activation=get_activation_from_str(self.activation))

    with tf.variable_scope('value_torso'):
      self.value_torso = snt.nets.MLP(
          value_torso_hidden_layer_sizes + [1],
          initializers=dict(w=glorot_uniform(seed),
                            b=initializers.init_ops.Constant(0)),
          activate_final=False,
          activation=get_activation_from_str(self.activation),
      )

  def _validate_observations(self, obs):
    for k in [
        'graph_features', 'node_mask', 'var_type_mask', 'constraint_type_mask',
        'obj_type_mask'
    ]:
      if k not in obs:
        raise Exception('%s not found in observation.' % k)

  def _encode(self, graph_features: gn.graphs.GraphsTuple, var_type_mask,
              constraint_type_mask, obj_type_mask):
    nodes = graph_features.nodes
    node_indices = gn.utils_tf.sparse_to_dense_indices(graph_features.n_node)
    l = [var_type_mask, constraint_type_mask, obj_type_mask]
    for i, mask in enumerate(l):
      mask = tf.reshape(mask, [-1, tf.shape(mask)[-1]])
      l[i] = tf.gather_nd(params=mask, indices=node_indices)
    var_type_mask, constraint_type_mask, obj_type_mask = l

    # TODO(arc): remove feature padding from nodes.
    nodes = tf.where(
        tf.equal(var_type_mask, 1), self._var_encode_net(nodes),
        tf.where(tf.equal(constraint_type_mask, 1),
                 self._constraint_encode_net(nodes),
                 self._obj_encode_net(nodes)))
    graph_features = graph_features.replace(nodes=nodes)
    graph_features = self._encode_net(graph_features)
    return graph_features

  def _convolve(self, graph_features: gn.graphs.GraphsTuple):
    for i in range(self.n_prop_layers):
      with tf.variable_scope('prop_layer_%d' % i):
        # one round of message passing
        new_graph_features = self._graphnet_models[i](graph_features)

        # residual connections
        graph_features = graph_features.replace(
            nodes=new_graph_features.nodes + graph_features.nodes,
            edges=new_graph_features.edges + graph_features.edges,
            globals=new_graph_features.globals + graph_features.globals)

    return graph_features

  def get_logits_and_next_state(self, step_type, _, obs, __):
    self._validate_observations(obs)
    graph_features = obs['graph_features']
    assert isinstance(graph_features, gn.graphs.GraphsTuple)
    # Run multiple rounds of graph convolutions
    graph_features = self._convolve(
        self._encode(graph_features, obs['var_type_mask'],
                     obs['constraint_type_mask'], obs['obj_type_mask']))
    # broadcast globals and attach them to node features
    graph_features = graph_features.replace(nodes=tf.concat([
        graph_features.nodes,
        gn.blocks.broadcast_globals_to_nodes(graph_features)
    ],
                                                            axis=-1))

    # get logits over nodes
    logits = self.policy_torso(graph_features.nodes)
    # remove the final singleton dimension
    logits = tf.squeeze(logits, axis=-1)
    log_vals = {}
    # record norm *before* adding -INF to invalid spots
    log_vals['opt/logits_norm'] = tf.linalg.norm(logits)

    indices = gn.utils_tf.sparse_to_dense_indices(graph_features.n_node)
    mask = obs['node_mask']
    logits = tf.scatter_nd(indices, logits, tf.shape(mask))
    logits = tf.where(tf.equal(mask, 1), logits, tf.fill(tf.shape(mask), -INF))
    return logits, self._dummy_state(tf.shape(step_type)[0]), log_vals

  def get_value(self, _, __, obs, ___):
    self._validate_observations(obs)
    with tf.variable_scope('value_network'):
      graph_features = self._convolve(
          self._encode(obs['graph_features'], obs['var_type_mask'],
                       obs['constraint_type_mask'], obs['obj_type_mask']))
      value = gn.blocks.NodesToGlobalsAggregator(
          tf.unsorted_segment_mean)(graph_features)
      value = tf.concat([value, graph_features.globals], axis=-1)
      return tf.squeeze(self.value_torso(value), axis=-1)

  def _dummy_state(self, bs):
    return tf.fill(tf.expand_dims(bs, 0), 0)

  def get_initial_state(self, bs):
    return self._dummy_state(bs)
