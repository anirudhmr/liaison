"""Adapted from github.com/tensorflow/models/blob/master/official/transformer/embedding_layer.py"""
import tensorflow as tf  # pylint: disable=g-bad-import-order


class Embedding(tf.layers.Layer):
  """Calculates input embeddings and pre-softmax linear with shared weights."""

  def __init__(self, vocab_size, hidden_size):
    """Specify characteristic parameters of embedding layer.
    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~32,000)
      hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
    """
    super().__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.built = False

  def build(self, x):
    if self.built:
      return

    with tf.variable_scope("embedding_and_softmax", reuse=tf.AUTO_REUSE):
      # Create and initialize weights. The random normal initializer was chosen
      # randomly, and works well.
      self.shared_weights = tf.get_variable(
          "weights", [self.vocab_size, self.hidden_size],
          initializer=tf.random_normal_initializer(0., self.hidden_size**-0.5))
    self.built = True

  def call(self, x):
    """Get token embeddings of x.
    Args:
      x: An int64 tensor with shape [batch_size, length]
    Returns:
      embeddings: float32 tensor with shape [batch_size, length, embedding_size]
      padding: float32 tensor with shape [batch_size, length] indicating the
        locations of the padding tokens in x.
    """
    with tf.name_scope("embedding"):
      # Create binary mask of size [batch_size, length]
      mask = tf.to_float(tf.not_equal(x, 0))
      embeddings = tf.gather(self.shared_weights, x)
      embeddings *= tf.expand_dims(mask, -1)
      # Scale embedding by the sqrt of the hidden size
      embeddings *= self.hidden_size**0.5
      return embeddings

  def linear(self, x):
    """Computes logits by running x through a linear layer.
    Args:
      x: A float32 tensor with shape [batch_size, length, hidden_size]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
    with tf.name_scope("presoftmax_linear"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      x = tf.reshape(x, [-1, self.hidden_size])
      logits = tf.matmul(x, self.shared_weights, transpose_b=True)

      return tf.reshape(logits, [batch_size, length, self.vocab_size])
