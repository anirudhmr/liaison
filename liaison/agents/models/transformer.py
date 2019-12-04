"""Graphnet based model."""

import graph_nets as gn
import numpy as np
import sonnet as snt
from liaison.agents.models.layers.attention import SelfAttention
from liaison.agents.models.utils import *
from sonnet.python.ops import initializers

INF = np.float32(1e9)


class Model:

  def __init__(self,
               seed,
               depth=6,
               node_embed_dim=32,
               embed_dim=128,
               policy_torso_hidden_layer_sizes=[64, 64],
               value_torso_hidden_layer_sizes=[64, 64],
               num_heads=8,
               key_dim=64,
               value_dim=64,
               action_spec=None):
    self.seed = seed
    self._depth = depth

    with tf.variable_scope('encode'):
      self._encode_layer = snt.nets.MLP(
          [node_embed_dim],
          activation=tf.nn.relu,
          initializers=dict(w=glorot_uniform(seed),
                            b=initializers.init_ops.Constant(0)),
          name='encode')
    with tf.variable_scope('embedding'):
      self._embedding_layer = snt.nets.MLP([embed_dim], activate_final=False)

    self._attn_layers = []
    self._ff_layers = []
    for i in range(depth):
      with tf.variable_scope('self_attention'):
        attn_layer = SelfAttention(node_embed_dim, num_heads, key_dim,
                                   value_dim)
        self._attn_layers.append(attn_layer)

      with tf.variable_scope('ff_layer'):
        ff_layer = snt.nets.MLP([4 * node_embed_dim, node_embed_dim],
                                activation=tf.nn.relu,
                                initializers=dict(
                                    w=glorot_uniform(seed),
                                    b=initializers.init_ops.Constant(0)),
                                activate_final=False)
        self._ff_layers.append(ff_layer)

    with tf.variable_scope('policy_torso'):
      self.policy_torso = snt.nets.MLP(
          policy_torso_hidden_layer_sizes + [1],
          initializers=dict(w=glorot_uniform(seed),
                            b=initializers.init_ops.Constant(0)),
          activate_final=False,
          activation=tf.nn.relu)

    with tf.variable_scope('value_torso'):
      self.value_torso = snt.nets.MLP(value_torso_hidden_layer_sizes + [1],
                                      initializers=dict(
                                          w=glorot_uniform(seed),
                                          b=initializers.init_ops.Constant(0)),
                                      activate_final=False,
                                      activation=tf.nn.relu)

  def _validate_observations(self, obs):
    for k in ['var_nodes', 'var_embeddings', 'mask']:
      if k not in obs:
        raise Exception('%s not found in observation.' % k)

  def _encode(self, obs):

    # var_nodes: [B, num_nodes, dim]
    var_nodes = obs['var_nodes']
    # var_nodes: [B, num_nodes, dim]
    var_embeddings = tf.cast(obs['var_embeddings'], tf.float32)

    var_embeddings = snt.BatchApply(self._embedding_layer)(var_embeddings)
    var_nodes = tf.concat([var_nodes, var_embeddings], -1)
    var_nodes = snt.BatchApply(self._encode_layer)(var_nodes)

    for i in range(self._depth):
      # layernorm(attn + residual)
      var_nodes = snt.LayerNorm()(self._attn_layers[i](var_nodes) + var_nodes)
      # layernorm(ff + residual)
      var_nodes = snt.BatchApply(self._ff_layers[i])(var_nodes) + var_nodes
      if i == self._depth - 1:
        var_nodes = snt.LayerNorm()(var_nodes)

    return var_nodes

  def get_logits_and_next_state(self, step_type, _, obs, __):
    self._validate_observations(obs)
    mask = obs['mask']
    node_embeds = self._encode(obs)
    # logits: [B, num_nodes, 1]
    logits = snt.BatchApply(self.policy_torso)(node_embeds)
    # remove the final singleton dimension
    logits = tf.squeeze(logits, axis=-1)

    log_vals = {}
    # record norm *before* adding -INF to invalid spots
    log_vals['opt/logits_norm'] = tf.linalg.norm(logits)

    logits = tf.where(tf.equal(mask, 1), logits, tf.fill(tf.shape(mask), -INF))
    return logits, self._dummy_state(tf.shape(step_type)[0]), log_vals

  def get_value(self, _, __, obs, ___):
    self._validate_observations(obs)
    with tf.variable_scope('value_network'):
      var_nodes = self._encode(obs)
      # [B, num_nodes, d] => [B, d]
      value = tf.reduce_sum(var_nodes, 1)
      return tf.squeeze(self.value_torso(value), axis=-1)

  def _dummy_state(self, bs):
    return tf.fill(tf.expand_dims(bs, 0), 0)

  def get_initial_state(self, bs):
    return self._dummy_state(bs)
