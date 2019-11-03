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
    "use_globals": True,
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


class Network:

  def __init__(
      self,
      seed,
      activation='relu',
      n_prop_layers=8,
      edge_embed_dim=32,
      node_embed_dim=32,
      global_embed_dim=32,
      node_hidden_layer_sizes=[64, 64],
      edge_hidden_layer_sizes=[64, 64],
  ):
    self.n_actions = n_actions
    self.activation = activation
    self.seed = seed
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

  def get_network(self):
    with tf.variable_scope('policy_network'):
      # global(node(edge))
      self.policy = gn.modules.GraphNetwork(
          edge_model_fn=lambda: self._edge_model,
          node_model_fn=lambda: self._node_model,
          global_model_fn=lambda: lambda x: x,
          node_block_opt=NODE_BLOCK_OPT,
          edge_block_opt=EDGE_BLOCK_OPT,
          global_block_opt=GLOBAL_BLOCK_OPT)
