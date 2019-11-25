"""Python RL Agent API."""

from __future__ import absolute_import, division, print_function

import abc
import collections
import random  # set seed

import liaison.utils as U
import numpy as np  # set seed
import six
import tensorflow as tf  # set seed
from liaison.agents.utils import *

# action, logits, next_state all have to be flat structures (list/nparrays/scalars.)
StepOutput = collections.namedtuple('StepOutput',
                                    ['action', 'logits', 'next_state'])


class Agent(object):
  """The base Agent class.
    The main API methods that users of this class need to know are:

        initial_state
        step
        update
    """

  def __new__(cls, *args, **kwargs):
    # We use __new__ since we want the env author to be able to
    # override __init__ without remembering to call super.
    return super(Agent, cls).__new__(cls)

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

    relative_norm = 0
    n_grads = 0
    for grad, var in clipped_grads_and_vars:
      if grad is not None:
        x = tf.linalg.norm(grad, name='grad_norm') / tf.linalg.norm(
            var, name='var_norm')
        x = tf.where(tf.is_nan(x), tf.zeros_like(x), x)
        relative_norm += x
        n_grads += 1

      if n_grads: relative_norm /= n_grads

    self._train_op = optimizer.apply_gradients(clipped_grads_and_vars,
                                               global_step=self.global_step)
    return {  # optimization related
        'opt/pre_clipped_grad_norm': global_norm,
        'opt/clipped_grad_norm': tf.linalg.global_norm(cli_grads, name='clipped_grad_norm'),
        'opt/lr': lr,
        'opt/weight_norm': tf.linalg.global_norm(variables, name='variable_norm'),
        'opt/relative_gradient_norm': relative_norm,
    }

  def _extract_logged_values(self, obs, reducer_fn):
    """
      If log_values found in obs then use reducer_fn and output as logs.
    """
    ret_d = dict()
    if 'log_values' in obs:
      d = tf.nest.map_structure(reducer_fn, obs['log_values'])
      ret_d = {}
      for k, v in d.items():
        assert not isinstance(v, dict)
        ret_d['log_values/' + k] = v
    return ret_d

  def initial_state(self, bs):
    """initial state of the agent.

    Args:
      bs: an int or placeholder for graph based agents.

    Returns:
      tensor corresponding to the initial state of the agent.
    """
    raise NotImplementedError(
        "initial_state is not implemented by the agent. ")

  def step(self, step_type, reward, obs, prev_state):
    """Step through and return an action.
    This function will only be called once for graph creation and
    the resulting graph will be run repeatedly for agent evaluation.

    All the below fields are expected to be batched in the first
    dimension.

    Args:
      step_type: Current steptype
      reward: Previous step reward.
      obs: Current Observations.
      prev_state: Prev agent state.

    Returns:
      StepOutput
      """
    raise NotImplementedError("Agent step function is not implemented.")

  def build_update_ops(self, step_outputs, prev_states, step_types, rewards,
                       observations, discounts):
    """Use trajectories collected to update the policy.

    This function will only be called once to create a TF graph which
    will be run repeatedly during training at the learner.

    All the arguments are tf placeholders (or nested structures of placeholders).

    Args:
      step_outputs: [T, B, ... ] of StepOutput structures.
      prev_states: [T + 1, B, ...] of agent prev states.
      step_types: [T + 1, B, ...] of stepTypes
      rewards: [T + 1, B, ...] of reward values
      observations: [T + 1, B, ...] of env observations.
      discounts: [T + 1, B] of discount values at each step.
    """
    raise NotImplementedError(
        "Agent build_update_ops function is not implemented.")

  def step_preprocess(self, step_type, reward, obs, prev_state):
    """The batch that is fed to step is preprocessed using this function.
      This will be called for every mini-batch. No tensorflow ops are
      allowed to be constructed here. The arguments are numpy or python objects.
    Example use-case:
      Graphnet de-padding and graph combining.

    Args:
      same as step but are numpy/python values instead of TF placeholders.
    """
    # override this for more interesting pre-processing steps.
    # call model.step_preprocess if that method exists
    if callable(getattr(self._model, 'step_preprocess', None)):
      return self._model.step_preprocess(step_type, reward, obs, prev_state)
    return step_type, reward, obs, prev_state

  def update_preprocess(self, step_outputs, prev_states, step_types, rewards,
                        observations, discounts):
    """
    The batch that is fed to build_update_ops is preprocessed using this function
    first.
    Example use-case:
      For graph-nets, use this to de-pad and merge the graphs
      to reduce the values that need to be copied to the GPUs.
      Also, it can be easy to implement some preprocess logic in
      numpy as opposed to tensorflow.

    Args:
      See self.build_update_ops function
    """

    # call model.update_preprocess if that method exists
    if callable(getattr(self._model, 'update_preprocess', None)):
      return self._model.update_preprocess(step_outputs, prev_states,
                                           step_types, rewards, observations,
                                           discounts)
    return step_outputs, prev_states, step_types, rewards, observations, discounts

  def update(self, sess, feed_dict):
    raise NotImplementedError("Agent update function is not implemented.")

  def set_seed(self, seed):
    self.seed = seed
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return args

  def update(self, sess, feed_dict):
    raise NotImplementedError("Agent update function is not implemented.")

  def set_seed(self, seed):
    self.seed = seed
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
