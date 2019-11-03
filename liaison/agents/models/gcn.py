"""Graphnet based model."""

import sonnet as snt
from sonnet.python.ops import initializers
from liaison.agents.models.utils import *
import graph_nets as gn


def make_mlp(layer_sizes, activation, activate_final, seed, layer_norm=False):
  mlp = snt.nets.MLP(
      layer_sizes,
      initializers=dict(
          w=glorot_uniform(seed),
          b=initializers.init_ops.Constant(0.1)),  # small bias initializer.
      activate_final=activate_final,
      activation=get_activation_from_str(activation),
  )
  if layer_norm:
    return snt.Sequential([mlp, snt.LayerNorm()])
  else:
    return mlp


EDGE_BLOCK_OPT = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": False,
}

NODE_BLOCK_OPT = {
    "use_received_edges": True,
    "use_sent_edges": True,
    "use_nodes": True,
    "use_globals": True,
}

GLOBAL_BLOCK_OPT = {
    "use_edges": False,
    "use_nodes": False,
    "use_globals": True,
}


class Model:

  def __init__(self,
               n_actions,
               seed,
               activation='relu',
               n_prop_layers=8,
               edge_embed_dim=32,
               node_embed_dim=32,
               global_embed_dim=32,
               node_hidden_layer_sizes=[64, 64],
               edge_hidden_layer_sizes=[64, 64],
               policy_torso_hidden_layer_sizes=[32, 32],
               value_torso_hidden_layer_sizes=[32, 32]):
    self.n_actions = n_actions
    self.activation = activation
    self.n_prop_layers = n_prop_layers
    self.seed = seed
    self.policy = None
    self.value = None
    with tf.variable_scope('edge_model'):
      self._edge_model = make_mlp(edge_hidden_layer_sizes,
                                  activation,
                                  True,
                                  seed,
                                  layer_norm=True)
    with tf.variable_scope('node_model'):
      self._node_model = make_mlp(node_hidden_layer_sizes,
                                  activation,
                                  True,
                                  seed,
                                  layer_norm=True)

    with tf.variable_scope('encode'):
      self._encode_net = gn.modules.GraphIndependent(
          edge_fn=lambda: snt.Linear(edge_embed_dim, name='edge_output'),
          node_fn=lambda: snt.Linear(node_embed_dim, name='node_output'),
          global_fn=lambda: snt.Linear(global_embed_dim, name='global_output'))

    with tf.variable_scope('graphnet_model'):
      # global(node(edge))
      self._graphnet = gn.modules.GraphNetwork(
          edge_model_fn=lambda: self._edge_model,
          node_model_fn=lambda: self._node_model,
          global_model_fn=lambda: lambda x:
          x,  # Don't summarize nodes/edges to the globals.
          node_block_opt=NODE_BLOCK_OPT,
          edge_block_opt=EDGE_BLOCK_OPT,
          global_block_opt=GLOBAL_BLOCK_OPT)

    with tf.variable_scope('policy_network'):
      self.policy_torso = snt.nets.MLP(
          policy_torso_hidden_layer_sizes + [n_actions],
          initializers=dict(w=glorot_uniform(seed),
                            b=initializers.init_ops.Constant(
                                0.1)),  # small bias initializer.
          activate_final=False,
          activation=get_activation_from_str(self.activation),
      )

    with tf.variable_scope('value_network'):
      self.value_torso = snt.nets.MLP(
          value_torso_hidden_layer_sizes + [1],
          initializers=dict(w=glorot_uniform(seed),
                            b=initializers.init_ops.Constant(
                                0.1)),  # small bias initializer.
          activate_final=False,
          activation=get_activation_from_str(self.activation),
      )

  def _dummy_state(self, bs):
    return tf.fill(tf.expand_dims(bs, 0), 0)

  def get_initial_state(self, bs):
    return self._dummy_state(bs)

  def _convolve(self, graph_features):
    """
      graph_features -> gn.graphs.GraphsTuple
    """
    assert isinstance(graph_features, gn.graphs.GraphsTuple)

    for i in range(self.n_prop_layers):
      with tf.variable_scope('prop_layer_%d' % i):
        # one round of message passing
        new_graph_features = self._graphnet(graph_features)

        # residual connections
        graph_features = graph_features.replace(
            nodes=new_graph_features.nodes + graph_features.nodes,
            edges=new_graph_features.edges + graph_features.edges,
            globals=new_graph_features.globals + graph_features.globals)

    return graph_features

  def get_logits_and_next_state(self, step_type, _, obs, __):

    if 'graph_features' not in obs:
      raise Exception('graph_features not found in observation.')

    # Run multiple rounds of graph convolutions
    graph_features = self._convolve(obs['graph_features'])

    # broadcast globals and attach them to node features
    graph_features = graph_features.replace(
        nodes=gn.broadcast_globals_to_nodes(graph_features))

    # get logits over nodes
    logits = self.policy_torso(graph_features.nodes)
    bs = tf.shape(step_type)[0]
    return logits, self._dummy_state(bs)

  def get_value(self, _, __, obs, ___):
    with tf.variable_scope('value_network'):
      graph_features = self._convolve(obs['graph_features'])
      value = gn.modules.NodesToGlobalsAggregator(graph_features)
      return self.value_torso(value)
