"""Python Supervised Agent API."""

import random  # set seed

import liaison.utils as U
import numpy as np  # set seed
import tensorflow as tf  # set seed
from liaison.agents.utils import get_decay_ops


class Agent(object):
  """The base Agent class.
  The main API methods that users of this class need to know are:
    build_update_ops
    update
  """

  def __new__(cls, *args, **kwargs):
    # We use __new__ since we want the child class author to be able to
    # override __init__ without remembering to call super.__init__.
    return super(Agent, cls).__new__(cls, *args, **kwargs)

  def _load_model(self, name, class_path, seed=None, **kwargs):
    # Loads a network based on the provided config.
    model_class = U.import_obj('Model', class_path)
    if seed is None:
      seed = self.seed
    with tf.variable_scope(name):
      self._model = model_class(seed=seed, **kwargs)
    return self._model

  def _lr_schedule(self):
    config = self.config
    return get_decay_ops(config.lr_init, config.lr_min,
                         config.lr_start_dec_step, config.lr_dec_steps,
                         config.lr_dec_val, config.lr_dec_approach)

  def _get_entropy_regularization_constant(self):
    config = self.config
    return get_decay_ops(config.ent_dec_init, config.ent_dec_min,
                         config.ent_start_dec_step, config.ent_dec_steps,
                         config.ent_dec_val, config.ent_dec_approach)

  def _init_optimizer(self, lr_op):
    return tf.train.AdamOptimizer(lr_op)

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
    clipped_grads_and_vars = list(zip(cli_grads, variables))

    # remove to stop large overhead
    # relative_norm = 0
    # n_grads = 0
    # for grad, var in clipped_grads_and_vars:
    #   if grad is not None:
    #     x = tf.linalg.norm(grad, name='grad_norm') / tf.linalg.norm(
    #         var, name='var_norm')
    #     x = tf.where(tf.is_nan(x), tf.zeros_like(x), x)
    #     relative_norm += x
    #     n_grads += 1

    #   if n_grads: relative_norm /= n_grads

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
        # 'opt/relative_gradient_norm': relative_norm,
    }

  def _extract_logged_values(self, obs):
    """
      If log_values found in obs then use reducer_fn and output as logs.
    """
    ret_d = dict()
    if 'log_values' in obs:
      ret_d = {}
      for k, v in obs['log_values'].items():
        ret_d['log_values/' + k] = v
    return ret_d

  def build_update_ops(self, obs, targets):
    """

    This function will only be called once to create a TF graph which
    will be run repeatedly during training at the learner.

    All the arguments are tf placeholders (or nested structures of placeholders).

    Args:
      obs: [B, ...] of observations.
      targets: [B, ...] of predictions
    """
    raise NotImplementedError(
        "Agent build_update_ops function is not implemented.")

  def update(self, sess, feed_dict, profile_kwargs):
    raise NotImplementedError("Agent update function is not implemented.")

  def set_seed(self, seed):
    self.seed = seed
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
