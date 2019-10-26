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

  def _load_model(self, class_path, seed=None, **kwargs):
    # Loads a network based on the provided config.
    model_class = U.import_obj('Model', class_path)
    if seed is None:
      seed = self.seed
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

  def update(self, sess, feed_dict):
    raise NotImplementedError("Agent update function is not implemented.")

  def set_seed(self, seed):
    self.seed = seed
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
