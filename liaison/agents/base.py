"""Python RL Agent API."""

from __future__ import absolute_import, division, print_function

import abc
import collections
import random  # set seed

import numpy as np  # set seed
import six
import tensorflow as tf  # set seed

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
    # override __init__ without remembering to call supGer.
    return super(Agent, cls).__new__(cls)

  def initial_state(self, batch_size):
    """initial state of the agent.

    Args:
      batch_size: an int or placeholder for graph based agents.

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

  def update(self, step_outputs, prev_states, step_types, rewards,
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
    raise NotImplementedError("Agent update function is not implemented.")

  def set_seed(self, seed):
    tf.random.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
