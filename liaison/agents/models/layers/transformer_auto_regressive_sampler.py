# Selects k of the inputs using a transformer model.
# Decoder generates queries which are dotted with the encoder keys
# to compute the attention weights and softmax over the input keys

from tqdm import tqdm

import graph_nets as gn
# Code adapted from https://github.com/Kyubyong/transformer/blob/master/model.py
import tensorflow as tf
from liaison.agents.utils import infer_shape
from liaison.utils import ConfigDict

from .transformer_utils import *


class Transformer:

  def __init__(self, **hp):
    self.hp = ConfigDict(hp)

  '''
    xs: tuple of
        x: int32 tensor. (N, T1, d)
        x_seqlens: int32 tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
    training: boolean.
  '''

  def encode(self, inp_embeddings, src_masks, training=True):
    '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
    '''
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
      enc = inp_embeddings  # enc -> (N, L1, d)

      ## Blocks
      for i in range(self.hp.num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
          # self-attention
          enc = multihead_attention(queries=enc,
                                    keys=enc,
                                    values=enc,
                                    key_masks=tf.logical_not(src_masks),
                                    num_heads=self.hp.num_heads,
                                    dropout_rate=self.hp.dropout_rate,
                                    training=training,
                                    causality=False)
          # feed forward
          enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
    memory = enc
    return memory

  def decode(self, dec, memory, src_masks, training=True):
    '''
        Args:
          memory: encoder outputs. (N, T1, d_model)

        Returns:
          logits: (N, T2, V). float32.
          y_hat: (N, T2). int32
          y: (N, T2). int32
        '''
    d_model = self.hp.d_model
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
      # Blocks
      for i in range(self.hp.num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
          # Masked self-attention (Note that causality is True at this time)
          dec = multihead_attention(queries=dec,
                                    keys=dec,
                                    values=dec,
                                    key_masks=None,
                                    num_heads=self.hp.num_heads,
                                    dropout_rate=self.hp.dropout_rate,
                                    training=training,
                                    causality=True,
                                    scope="self_attention")

          # Vanilla attention
          dec = multihead_attention(queries=dec,
                                    keys=memory,
                                    values=memory,
                                    key_masks=tf.logical_not(src_masks),
                                    num_heads=self.hp.num_heads,
                                    dropout_rate=self.hp.dropout_rate,
                                    training=training,
                                    causality=False,
                                    scope="vanilla_attention")
          ### Feed Forward
          dec = ff(dec, num_units=[self.hp.d_ff, d_model])

    return dec
