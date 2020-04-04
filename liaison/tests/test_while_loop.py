import sys

import tensorflow as tf


class model(object):

  def __init__(self):
    story_len = tf.constant(7)
    starter = tf.constant(0)
    z = []

    def body(hops):
      with tf.control_dependencies([tf.print(hops, output_stream=sys.stdout)]):
        return hops + 1

    def condition(hops):
      return hops < story_len

    self.gate_index = tf.while_loop(condition, body, [starter])

  def step(self, sess):
    print(sess.run([self.gate_index]))


with tf.Session() as sess:
  while_loop = model()
  sess.run(tf.initialize_all_variables())
  while_loop.step(sess)
