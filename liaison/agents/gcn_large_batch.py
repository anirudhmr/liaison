import collections

import liaison.utils as U
import numpy as np  # set seed
import six
import tensorflow as tf  # set seed
import tree as nest
from liaison.agents.gcn import Agent as GCNAgent
from liaison.agents.utils import *


class Agent(GCNAgent):

  def __init__(self, *args, apply_grads_every=1, **kwargs):
    super(Agent, self).__init__(*args, **kwargs)
    self._shadow_vars = None
    # shadow assignments assign value of shadow to main variable
    self._shadow_assignments = None
    self._apply_grads_every = apply_grads_every

  def _create_shadow_vars(self, variables):
    assert self._shadow_vars is None
    # Note that the shadow variables are out of the scope and cannot
    # be published.
    l = []
    assignments = []
    with tf.variable_scope('shadow_vars'):
      for var in variables:
        shadow_var = tf.Variable(name=f'{var.name.replace(":", "_")}',
                                 initial_value=var,
                                 trainable=False)
        l.append(shadow_var)
        # assign shadow var to the main policy variable.
        assignments.append(
            var.assign(shadow_var,
                       name=f'assign_{shadow_var.name.replace(":", "_")}',
                       read_value=False))

    self._shadow_vars = l
    self._shadow_assignments = assignments

  def _optimize(self, loss):
    config = self.config
    self.global_step = tf.train.get_or_create_global_step()
    lr = self._lr_schedule()
    optimizer = self._init_optimizer(lr)

    # get clipped gradients
    grads, variables = zip(*optimizer.compute_gradients(loss))
    global_norm = tf.linalg.global_norm(grads, 'grad_norm')
    if config.grad_clip > 0:
      # clip_by_global_norm does t_list[i] <- t_list[i] * clip_norm / max(global_norm, clip_norm)
      cli_grads, _ = tf.clip_by_global_norm(grads,
                                            config.grad_clip,
                                            name='clip_by_global_norm')
    else:
      cli_grads = grads

    self._create_shadow_vars(variables)
    # always apply gradients to the shadow variables.
    clipped_grads_and_vars = list(zip(cli_grads, self._shadow_vars))

    self._train_op = optimizer.apply_gradients(clipped_grads_and_vars,
                                               global_step=self.global_step)

    return {  # optimization related
        'opt/pre_clipped_grad_norm':
        global_norm,
        'opt/clipped_grad_norm':
        tf.linalg.global_norm(cli_grads, name='clipped_grad_norm'),
        'opt/lr':
        lr,
        'opt/weight_norm':
        tf.linalg.global_norm(variables, name='variable_norm'),
    }

  def update(self, sess, feed_dict, profile_kwargs):
    """profile_kwargs pass to sess.run for profiling purposes."""
    step = self.global_step.eval(sess)
    if step > 0 and step % self._apply_grads_every == 0:
      sess.run(self._shadow_assignments)

    _, vals = sess.run([self._train_op, self._logged_values],
                       feed_dict=feed_dict,
                       **profile_kwargs)
    return vals
