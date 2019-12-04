"""Adapted from github.com/tensorflow/models/blob/master/official/transformer/model/attention_layer.py"""
import tensorflow as tf


class Attention(tf.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(self, hidden_size, num_heads, k_size, v_size):
    if hidden_size % num_heads != 0:
      raise ValueError("Hidden size must be evenly divisible by the number of "
                       "heads.")

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    # Layers for linearly projecting the queries, keys, and values.
    self.q_dense_layer = tf.layers.Dense(k_size * num_heads,
                                         use_bias=False,
                                         name="q")
    self.k_dense_layer = tf.layers.Dense(k_size * num_heads,
                                         use_bias=False,
                                         name="k")
    self.v_dense_layer = tf.layers.Dense(v_size * num_heads,
                                         use_bias=False,
                                         name="v")

    self.output_dense_layer = tf.layers.Dense(hidden_size,
                                              use_bias=False,
                                              name="output_transform")

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.
    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.
    Args:
      x: A tensor with shape [batch_size, length, hidden_size]
    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, -1])

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3])

  def combine_heads(self, x):
    """Combine tensor that has been split.
    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]
    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x,
                       [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
      return tf.reshape(x, [batch_size, length, -1])

  def call(self, x, y):
    """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    q = self.q_dense_layer(x)
    k = self.k_dense_layer(y)
    v = self.v_dense_layer(y)

    # Split q, k, v into heads.
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    # Scale q to prevent the dot product between q and k from growing too large.
    depth = (self.hidden_size // self.num_heads)
    q *= depth**-0.5

    # Calculate dot product attention
    logits = tf.matmul(q, k, transpose_b=True)
    weights = tf.nn.softmax(logits, name="attention_weights")
    # if self.train:
    #   weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
    attention_output = tf.matmul(weights, v)

    # Recombine heads --> [batch_size, length, num_heads * value_dim]
    attention_output = self.combine_heads(attention_output)

    # Run the combined outputs through another linear projection layer.
    # [batch_size, length, hidden_size]
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self, x):
    return super(SelfAttention, self).call(x, x)
