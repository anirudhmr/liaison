"""Graphnet based model."""
import graph_nets as gn
import liaison.utils as U
import sonnet as snt
from liaison.agents.models.gcn_rins import make_mlp
from liaison.agents.models.layers.transformer_auto_regressive_sampler import \
    Transformer
from liaison.agents.models.layers.transformer_utils import *
from liaison.agents.models.utils import *
from liaison.agents.utils import infer_shape, sample_from_logits
from liaison.utils import ConfigDict
from sonnet.python.ops import initializers
from tqdm import tqdm


class Model:

  def __init__(self, seed, model_kwargs, action_spec=None, **kwargs):
    self.seed = seed
    self.k = action_spec.shape[-1]
    self.config = ConfigDict(kwargs)
    with tf.variable_scope('gcn_model'):
      self._gcn_model = U.import_obj('Model', model_kwargs['class_path'])(
          seed=seed, **model_kwargs)

    with tf.variable_scope('transformer'):
      self._trans = Transformer(**self.config)

    with tf.variable_scope('enc_node_embeds'):
      self._node_emb_mlp = make_mlp([self.config.d_model] * 2,
                                    'relu',
                                    activate_final=False,
                                    seed=seed,
                                    layer_norm=False)

  def compute_graph_embeddings(self, obs):
    return self._gcn_model.compute_graph_embeddings(obs)

  def get_actions(self, graph_embeddings, obs, sampled_actions=None):
    # If actions ar eprovided, then sampling is ommitted.
    """
      Difference between src_masks, node_mask and action_mask
        src_masks: Indicates that certain node inputs to the encoder
                   are from padding and should not be considered.
        node_masks: Sampling should happen only at these nodes during
                    decoding phase.
        action_mask: Masks where the previous action has taken place.
    """
    xs, n_node = self._gcn_model.get_node_embeddings(
        obs, graph_embeddings)  # (N, L1, d)
    node_mask = tf.reshape(obs['node_mask'], infer_shape(xs)[:-1])
    node_mask = tf.cast(node_mask, tf.bool)
    if sampled_actions is not None:
      # Remove some of the samples from the batch to match the size of the
      # actions.
      # This happens because the graph_embeddings are calculated for T + 1
      # observations whereas actions are only avaiable for T steps.
      # (T + 1) * B -> T * B
      xs = xs[:infer_shape(sampled_actions)[0]]
      n_node = n_node[:infer_shape(sampled_actions)[0]]
      node_mask = node_mask[:infer_shape(sampled_actions)[0]]
    xs = snt.BatchApply(self._node_emb_mlp)(xs)  # (N, L1, d_model)
    # src_masks -> (N, T1)
    memory, src_masks = self._trans.encode(xs, n_node)

    N = infer_shape(xs)[0]
    # Feed in zero embedding as the start sentinel.
    _decoder_inputs = tf.zeros((N, 1) + (infer_shape(xs)[-1], ),
                               tf.float32)  # (N, 1, d)
    logitss = []
    actions = []

    for i in tqdm(range(self.k)):
      dec = self._trans.decode(_decoder_inputs, memory, src_masks,
                               True)  # (N, T1, d_model)

      with tf.variable_scope('attn_head', reuse=tf.AUTO_REUSE):
        # Q -> (N, d_model)
        Q = tf.layers.dense(dec[:, -1], self.config.d_model, use_bias=True)
        # Q -> (N, 1, d_model)
        Q = tf.expand_dims(Q, 1)

      # dot product
      outputs = tf.matmul(Q, tf.transpose(xs, [0, 2, 1]))  # (N, 1, T1)
      outputs = tf.squeeze(outputs, 1)  # (N, T1)

      # scale
      outputs /= (Q.get_shape().as_list()[-1]**0.5)

      indices = gn.utils_tf.sparse_to_dense_indices(n_node)
      logits = tf.where(node_mask, outputs,
                        tf.fill(tf.shape(node_mask), np.float32(-1e9)))

      logitss.append(logits)
      if sampled_actions is None:
        act = sample_from_logits(logits, self.seed)  # (N,)
      else:
        act = sampled_actions[:, i]
      actions.append(act)

      # update node_masks to remove the current selected node for the next
      # decoding iteration.
      indices = tf.stack([tf.range(N), act], axis=-1)
      action_mask = tf.scatter_nd(indices, tf.ones((N, ), tf.bool),
                                  infer_shape(node_mask))
      node_mask = tf.logical_and(node_mask, tf.logical_not(action_mask))

      embs = tf.gather_nd(xs, indices)  # (N, d)
      embs = tf.expand_dims(embs, 1)  # (N, 1, d)
      _decoder_inputs = tf.concat((_decoder_inputs, embs), 1)

    logits = tf.stack(logitss, axis=1)  # (N, T2, T1)
    actions = tf.stack(actions, axis=1)  # [N, T2]
    return logits, actions

  def get_value(self, graph_embeddings):
    # Give this the optimal solution as well in the inputs?
    return self._gcn_model.get_value(graph_embeddings)

  def dummy_state(self, bs):
    return tf.fill(tf.expand_dims(bs, 0), 0)

  def get_initial_state(self, bs):
    return self.dummy_state(bs)
