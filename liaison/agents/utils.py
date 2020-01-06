import graph_nets as gn
import tensorflow as tf


def sample_from_logits(logits, seed):
  return tf.squeeze(tf.random.categorical(logits, 1, dtype=tf.int32,
                                          seed=seed),
                    axis=-1)


def compute_entropy(logits):
  """Calculate entropy of distribution."""
  # https://blog.feedly.com/tricks-of-the-trade-logsumexp/
  # Also see softmax implementation here:
  # https://github.com/tensorflow/tensorflow/blob/1cf0898dd4331baf93fe77205550f2c2e6c90ee5/tensorflow/python/keras/activations.py#L44
  return tf.reduce_sum(-tf.nn.softmax(logits) * tf.nn.log_softmax(logits), -1)


def infer_shape(x):
  """
    x -> Tensor
    Wherever possible use the static dimension,
    else source it dynamically

    Returns:
      Tensor with rank 1 and unknown dims if rank(x) is not known
      List of integers or tensor values if rank(x) is known.
  """
  # If unknown rank, return dynamic shape
  if x.shape.dims is None:
    return tf.shape(x)

  static_shape = x.shape.as_list()
  dynamic_shape = tf.shape(x)

  ret = []
  for i in range(len(static_shape)):
    dim = static_shape[i]
    if dim is None:
      dim = dynamic_shape[i]
    ret.append(dim)

  return ret


def merge_first_two_dims(tensor, validate=True):
  shape = infer_shape(tensor)
  if isinstance(shape, list):
    if len(shape) < 2:
      if validate:
        raise Exception('less than two dimensions found')
      else:
        print('Warning: Less than two dimensions found while merging')
        return tensor

    shape[0] *= shape[1]
    shape.pop(1)
  else:
    # dynamic shape
    shape = tf.concat([[shape[0] * shape[1]], shape[2:]], axis=0)
  return tf.reshape(tensor, shape)


def get_decay_ops(init_val,
                  min_val,
                  start_decay_step,
                  decay_steps,
                  dec_val,
                  dec_approach,
                  global_step=None):
  """Decay value starting from init_val by dec_val every decay_steps."""
  if global_step is None:
    global_step = tf.train.get_or_create_global_step()
  val_gstep = global_step - start_decay_step

  init_val = float(init_val)
  min_val = float(min_val)
  dec_val = float(dec_val)

  def f1():
    return tf.constant(init_val)

  def f2():
    if dec_approach == 'exponential':
      return tf.train.exponential_decay(init_val, val_gstep, decay_steps,
                                        dec_val)
    elif dec_approach == 'linear':
      return tf.train.polynomial_decay(init_val, val_gstep, decay_steps,
                                       min_val)
    elif dec_approach == 'constant':
      return tf.constant(init_val)
    else:
      raise Exception('Unknown lr decay approach: %s' % dec_approach)

  op = tf.cond(tf.less(global_step, start_decay_step), f1, f2)
  return tf.maximum(op, min_val)


def flatten_graphs(graph_features):
  """
    Flatten graphs. Remove padding.
    Args:
      graph_features: gn.graphs.GraphsTuple.
      B is batch size
      M_N is the max # of node (graphs with < M_N nodes use padding)
      M_E max # of edges.
      graph_features.nodes    => [B, M_N, ...]
      graph_features.edges    => [B, M_E, ...]
      graph_features.senders  => [B, M_E]
      graph_features.receivers=> [B, M_E]
      graph_features.n_node   => [B]
      graph_features.n_edge   => [B]
      graph_features.globals  => [B, ...]
    Returns:
      graph_features: gn.graphs.GraphsTuple
      Let S_N = sum(graph_features.n_node)
      Let S_E = sum(graph_features.n_edge)
      graph_features.nodes    => [S_N, ...]
      graph_features.edges    => [S_E, ...]
      graph_features.senders  => [S_E]
      graph_features.receivers=> [S_E]
      graph_features.n_node   => [B]
      graph_features.n_edge   => [B]
      graph_features.globals  => [B, ...]
  """
  node_indices = gn.utils_tf.sparse_to_dense_indices(graph_features.n_node)
  edge_indices = gn.utils_tf.sparse_to_dense_indices(graph_features.n_edge)
  graph_features = graph_features.replace(
      nodes=tf.gather_nd(params=graph_features.nodes, indices=node_indices),
      edges=tf.gather_nd(params=graph_features.edges, indices=edge_indices),
      senders=tf.gather_nd(params=graph_features.senders,
                           indices=edge_indices),
      receivers=tf.gather_nd(params=graph_features.receivers,
                             indices=edge_indices))
  return gn.utils_tf.stop_gradient(graph_features)
