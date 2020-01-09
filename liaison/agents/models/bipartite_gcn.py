import numpy as np
import tensorflow as tf
import tensorflow.keras as K


class BipartiteGraphConvolution(K.Model):
  """
    Partial bipartite graph convolution (either left-to-right or right-to-left).
  """

  def __init__(self,
               emb_size,
               activation,
               initializer,
               right_to_left=False,
               memory_hack=False, sum_aggregation=True):
    """
      For memory_hack see issue https://github.com/ds4dm/learn2branch/issues/4
    """
    super().__init__()
    self.emb_size = emb_size
    self.activation = activation
    self.initializer = initializer
    self.right_to_left = right_to_left
    self.memory_hack = memory_hack

    # feature layers
    self.feature_module_left = K.Sequential([
        K.layers.Dense(units=self.emb_size,
                       activation=None,
                       use_bias=True,
                       kernel_initializer=self.initializer)
    ])
    self.feature_module_edge = K.Sequential([
        K.layers.Dense(units=self.emb_size,
                       activation=None,
                       use_bias=False,
                       kernel_initializer=self.initializer)
    ])
    self.feature_module_right = K.Sequential([
        K.layers.Dense(units=self.emb_size,
                       activation=None,
                       use_bias=False,
                       kernel_initializer=self.initializer)
    ])

    # the convolution model (edge_model)
    self.feature_module_final = K.Sequential([
        K.layers.Activation(self.activation),
        K.layers.Dense(units=self.emb_size,
                       activation=None,
                       kernel_initializer=self.initializer)
    ])

    # output_layers
    self.output_module = K.Sequential([
        K.layers.Dense(units=self.emb_size,
                       activation=None,
                       kernel_initializer=self.initializer),
        K.layers.Activation(self.activation),
        K.layers.Dense(units=self.emb_size,
                       activation=None,
                       kernel_initializer=self.initializer),
    ])
    self.sum_aggregation = sum_aggregation

  def call(self, inputs):
    """
      Perfoms a partial graph convolution on the given bipartite graph.
      Inputs
      ------
      left_features: 2D float tensor
        Features of the left-hand-side nodes in the bipartite graph
      edge_indices: 2D int tensor
        Edge indices in left-right order
      edge_features: 2D float tensor
        Features of the edges
      right_features: 2D float tensor
        Features of the right-hand-side nodes in the bipartite graph
    """
    left_features, edge_indices, edge_features, right_features = inputs

    if self.right_to_left:
      scatter_dim = 0
      prev_features = left_features
    else:
      scatter_dim = 1
      prev_features = right_features
    scatter_out_size = tf.shape(prev_features)[0]

    left_features = tf.gather(self.feature_module_left(left_features),
                              axis=0,
                              indices=edge_indices[0])
    edge_features = self.feature_module_edge(edge_features)
    right_features = tf.gather(self.feature_module_right(right_features),
                               axis=0,
                               indices=edge_indices[1])

    # compute joint features
    if memory_hack:
      joint_features = left_features + edge_features + right_features
    else:
      joint_features = tf.concat(
          [left_features, edge_features, right_features], axis=-1)

    # perform convolution
    joint_features = self.feature_module_final(joint_features)
    conv_output = tf.scatter_nd(updates=joint_features,
                                indices=tf.expand_dims(
                                    edge_indices[scatter_dim], axis=1),
                                shape=[scatter_out_size, self.emb_size])
    # mean convolution
    neighbour_count = tf.scatter_nd(
        updates=tf.ones(shape=[tf.shape(edge_indices)[1], 1],
                        dtype=tf.float32),
        indices=tf.expand_dims(edge_indices[scatter_dim], axis=1),
        shape=[scatter_out_size, 1])
    neighbour_count = tf.where(tf.equal(neighbour_count, 0),
                               tf.ones_like(neighbour_count),
                               neighbour_count)  # NaN safety trick
    conv_output = conv_output / neighbour_count

    # apply final module
    output = self.output_module(
        tf.concat([
            conv_output,
            prev_features,
        ], axis=1))

    return output


class Model(K.Model):
  """
    Our bipartite Graph Convolutional neural Network (GCN) model.
    Assumes that the left side nodes are senders and right are receivers.
  """

  def __init__(self):
    super().__init__()

    self.emb_size = 64
    self.cons_nfeats = 5
    self.edge_nfeats = 1
    self.var_nfeats = 19

    self.activation = K.activations.relu
    self.initializer = K.initializers.Orthogonal()

    # CONSTRAINT EMBEDDING
    self.cons_embedding = K.Sequential([
        K.layers.Dense(units=self.emb_size,
                       activation=self.activation,
                       kernel_initializer=self.initializer) for _ in range(2)
    ])

    # VARIABLE EMBEDDING
    self.var_embedding = K.Sequential([
        K.layers.Dense(units=self.emb_size,
                       activation=self.activation,
                       kernel_initializer=self.initializer) for _ in range(2)
    ])

    # GRAPH CONVOLUTIONS
    self.conv_v_to_c = BipartiteGraphConvolution(self.emb_size,
                                                 self.activation,
                                                 self.initializer,
                                                 right_to_left=True)
    self.conv_c_to_v = BipartiteGraphConvolution(self.emb_size,
                                                 self.activation,
                                                 self.initializer)

    # OUTPUT
    self.policy_torso = K.Sequential([
        K.layers.Dense(units=self.emb_size,
                       activation=self.activation,
                       kernel_initializer=self.initializer),
        K.layers.Dense(units=1,
                       activation=None,
                       kernel_initializer=self.initializer,
                       use_bias=False),
    ])

  def _validate_observations(self, obs):
    if 'graph_features' not in obs:
      raise Exception('graph_features not found in observation.')
    elif 'node_mask' not in obs:
      raise Exception('node_mask not found in observation.')
    elif 'left_feature_dim' not in obs:
      raise Exception('left_feature_dim not found in observation.')
    elif 'right_feature_dim' not in obs:
      raise Exception('right_feature_dim not found in observation.')

  def get_logits_and_next_state(self, step_type, _, obs, __):
    self._validate_observations(obs)
    log_vals = {}
    graph_features = obs['graph_features']
    assert isinstance(graph_features, gn.graphs.GraphsTuple)

    # gather receiver node features as right features
    variable_features = tf.gather(graph_features.nodes,
                                  axis=0,
                                  indices=graph_features.receivers)
    variable_features = variable_features[:, 0:obs['right_feature_dim']]

    # gather sender node features as left features
    constraint_features = tf.gather(graph_features.nodes,
                                    axis=0,
                                    indices=graph_features.senders)
    constraint_features = constraint_features[:, 0:obs['left_feature_dim']]

    # EMBEDDINGS
    variable_features = self.var_embedding(variable_features)
    constraint_features = self.cons_embedding(constraint_features)

    # GRAPH CONVOLUTIONS
    constraint_features = self.conv_v_to_c(
        (constraint_features, (graph_features.senders,
                               graph_features.receivers), graph_features.edges,
         variable_features))
    constraint_features = self.activation(constraint_features)

    variable_features = self.conv_c_to_v(
        (constraint_features, (graph_features.senders,
                               graph_features.receivers), graph_features.edges,
         variable_features))
    variable_features = self.activation(variable_features)

    # OUTPUT
    output = self.policy_torso(variable_features)
    output = tf.reshape(output, [1, -1])
    return output

  def call(self, *args, **kwargs):
    return self.get_logits_and_next_state(*args, **kwargs)
