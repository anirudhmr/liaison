"""Graphnet based model."""
import tempfile

import graph_nets as gn
import liaison.utils as U
import matplotlib as mpl
import matplotlib.pyplot as plt
import sonnet as snt
from liaison.agents.models.gcn_rins import make_mlp
from liaison.agents.models.layers.transformer_auto_regressive_sampler import \
    Transformer
from liaison.agents.models.layers.transformer_utils import *
from liaison.agents.models.utils import *
from liaison.agents.utils import infer_shape, sample_from_logits
from liaison.env import StepType
from liaison.utils import ConfigDict
from sklearn.manifold import TSNE
from sonnet.python.ops import initializers
from tqdm import tqdm

mpl.use('Agg')
plt.style.use('seaborn')


class Model:

  def __init__(self, seed, model_kwargs, action_spec=None, **kwargs):
    self.seed = seed
    self.k = action_spec.shape[-1]
    self.config = ConfigDict(kwargs)
    with tf.variable_scope('gcn_model'):
      self._gcn_model = U.import_obj('Model', model_kwargs['class_path'])(seed=seed,
                                                                          **model_kwargs)

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

  def get_actions(self,
                  graph_embeddings,
                  obs,
                  step_types=None,
                  sampled_actions=None,
                  log_features=False):
    # step_types: [T + 1, B] of stepTypes
    # If actions ar eprovided, then sampling is ommitted.
    """
      Difference between src_masks, node_mask and action_mask
        src_masks: Indicates that certain node inputs to the encoder
                   are from padding and should not be considered.
        node_masks: Sampling should happen only at these nodes during
                    decoding phase.
        action_mask: Masks where the previous action has taken place.
    """
    xs, n_node = self._gcn_model.get_node_embeddings(obs, graph_embeddings)  # (N, L1, d)
    node_mask = tf.reshape(obs['node_mask'], infer_shape(xs)[:-1])
    node_mask = tf.cast(node_mask, tf.bool)
    n_actions = tf.reshape(obs['n_actions'], (infer_shape(xs)[0], ))
    if sampled_actions is not None:
      # Remove some of the samples from the batch to match the size of the
      # actions.
      # This happens because the graph_embeddings are calculated for T + 1
      # observations whereas actions are only avaiable for T steps.
      # (T + 1) * B -> T * B
      t_times_b = infer_shape(sampled_actions)[0]
      xs = xs[:t_times_b]
      n_node = n_node[:t_times_b]
      node_mask = node_mask[:t_times_b]
      n_actions = n_actions[:t_times_b]
    N = infer_shape(xs)[0]
    T1 = infer_shape(xs)[1]  # T1 and L1 used interchangebly.

    # compute src_mask to remove padding nodes interfering.
    indices = gn.utils_tf.sparse_to_dense_indices(n_node)
    # src_masks -> (N, L1)
    src_masks = tf.scatter_nd(indices, tf.ones(infer_shape(indices)[:1], tf.bool),
                              infer_shape(xs)[:2])

    if sampled_actions is not None:
      log_features = dict()
      log_features.update(dict(xs=xs, src_masks=src_masks, step_types=step_types[:-1]))
      log_features.update(dict(actions=sampled_actions))

    xs = snt.BatchApply(self._node_emb_mlp)(xs)  # (N, L1, d_model)
    memory = xs

    d = self.config.d_model
    # Feed in zero embedding as the start sentinel.
    _decoder_inputs = tf.zeros((N, 1, d), tf.float32)  # (N, 1, d)

    def cond_fn(i, *_):
      # until all samples have n_actions sampled.
      return i < tf.reduce_max(n_actions)

    def body_fn(i, decoder_inputs, logitss, actions, node_mask):
      dec = self._trans.decode(decoder_inputs, memory, src_masks, True)  # (N, T1, d_model)

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

      # [N, T1]
      logits = tf.where(node_mask, outputs, tf.fill(tf.shape(node_mask), np.float32(-1e9)))

      if sampled_actions is None:
        act = sample_from_logits(logits, self.seed)  # (N,)
      else:
        act = sampled_actions[:, i]

      logitss = tf.concat([logitss, tf.expand_dims(logits, axis=1)], axis=1)
      actions = tf.concat([actions, tf.expand_dims(act, axis=1)], axis=-1)

      # update node_masks to remove the current selected node for the next
      # decoding iteration.
      indices = tf.stack([tf.range(N), act], axis=-1)
      action_mask = tf.scatter_nd(indices, tf.ones((N, ), tf.bool), infer_shape(node_mask))
      node_mask = tf.logical_and(node_mask, tf.logical_not(action_mask))

      embs = tf.gather_nd(xs, indices)  # (N, d)
      embs = tf.expand_dims(embs, 1)  # (N, 1, d)
      decoder_inputs = tf.concat((decoder_inputs, embs), 1)
      return i + 1, decoder_inputs, logitss, actions, node_mask

    i = tf.constant(0)
    logitss = tf.constant([], shape=(N, 0, T1), dtype=tf.float32)
    actions = tf.constant([], shape=(N, 0), dtype=tf.int32)
    i, _, logits, actions, _ = tf.while_loop(cond_fn,
                                             body_fn,
                                             [i, _decoder_inputs, logitss, actions, node_mask],
                                             shape_invariants=[
                                                 i.get_shape(),
                                                 tf.TensorShape([N, None, d]),
                                                 tf.TensorShape([N, None, T1]),
                                                 tf.TensorShape([N, None]),
                                                 node_mask.get_shape(),
                                             ])

    # Finally logits -> (N, T2, T1)
    # Finally actions -> (N, T2)
    if log_features:
      return logits, actions, log_features
    return logits, actions

  def log_features(self, features, step, loggers):
    # log the features collected in self._log_features
    # step_types -> (T, B)
    step_types = features['step_types']
    xs = features['xs']
    # xs -> (T, B, N_max, d)
    xs = np.reshape(xs, step_types.shape + xs.shape[1:])
    src_masks = features['src_masks']
    # src_masks -> (T, B, N_max)
    src_masks = np.reshape(src_masks, step_types.shape + src_masks.shape[1:])
    actions = features['actions']
    # actions -> (T, B, N_acts)
    actions = np.reshape(features['actions'], step_types.shape + actions.shape[1:])

    N_FIGS = 4
    collected_indices = []
    for t, step_type in enumerate(step_types):
      if len(collected_indices) >= N_FIGS:
        break
      if np.any(step_type == StepType.FIRST):
        for b, s in enumerate(step_type):
          if s == StepType.FIRST:
            collected_indices.append((t, b))

    collected_indices = collected_indices[:N_FIGS]

    fig, axes = plt.subplots(ncols=len(collected_indices), figsize=[8 * len(collected_indices), 8])
    if len(collected_indices) == 1:
      axes = [axes]

    for (t, b), ax in zip(collected_indices, axes):
      x = xs[t, b][src_masks[t, b]]
      points = TSNE(n_components=2).fit_transform(x)
      x, y = points[:, 0], points[:, 1]
      colors = ['black'] * len(x)
      for act in actions[t, b]:
        colors[act] = 'red'
      ax.scatter(x, y, c=colors)

    with tempfile.SpooledTemporaryFile(mode='rw+b') as f:
      fig.savefig(f, format='png')
      for logger in loggers:
        f.seek(0)
        logger.write(f, step=step, fname=f'graph_embedding_tsne_{step}.png')

  def get_value(self, graph_embeddings, obs):
    # Give this the optimal solution as well in the inputs?
    if self.config.use_mlp_value_func:
      xs, _ = self._gcn_model.get_node_embeddings(obs, graph_embeddings)  # (N, L1, d)
      xs = tf.reshape(xs, [infer_shape(xs)[0], -1])

      with tf.variable_scope('value_network'):
        self.value = snt.nets.MLP([32, 32, 1],
                                  initializers=dict(w=glorot_uniform(self.seed),
                                                    b=initializers.init_ops.Constant(0.0)),
                                  activate_final=False,
                                  activation=get_activation_from_str('relu'))
      return tf.squeeze(self.value(xs), axis=-1)
    else:
      return self._gcn_model.get_value(graph_embeddings)

  def dummy_state(self, bs):
    return tf.fill(tf.expand_dims(bs, 0), 0)

  def get_initial_state(self, bs):
    return self.dummy_state(bs)
