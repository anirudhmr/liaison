import functools

import numpy as np

import tensorflow as tf
from absl import logging
from liaison.agents import BaseAgent, StepOutput, utils, vtrace_ops
from liaison.agents.losses.vtrace import Loss as VTraceLoss
from liaison.agents.utils import *
from liaison.env import StepType
from liaison.utils import ConfigDict
from tensorflow.contrib.framework import nest

_ACTION_SPEC_UNBOUNDED = 'action spec `{}` is not correctly bounded.'
_DTYPE_NOT_INTEGRAL = '`dtype` must be integral, got {}.'
_SHAPE_NOT_SPECIFIED = 'action spec `shape` must be fully specified, got {}.'


class Agent(BaseAgent):

  def __init__(self, name, action_spec, seed, model=None, **kwargs):

    self.set_seed(seed)
    self.config = ConfigDict(**kwargs)
    self._name = name
    self._action_spec = action_spec
    self._load_model(name, action_spec=action_spec, **(model or {}))

  def initial_state(self, bs):
    return self._model.get_initial_state(bs)

  def step(self, step_type, reward, obs, prev_state):
    """Step through and return an action.
    This function will only be called once for graph creation and
    the resulting graph will be run repeatedly for agent evaluation.

    All the below fields are expected to be batched in the first
    dimension. (No time dimension)

    Args:
      step_type: [B,] Current steptype
      reward: [B,] Previous step reward.
      obs: Current Observations.
      prev_state: Prev agent state.

    Returns:
      StepOutput
    """
    with tf.variable_scope(self._name):
      logits, next_state, _ = self._model.get_logits_and_next_state(
          step_type, reward, obs, prev_state)
      action = sample_from_logits(logits, self.seed)
      return StepOutput(action, logits, next_state)

  def build_update_ops(self, step_outputs, prev_states, step_types, rewards,
                       observations, discounts):
    """Use trajectories collected to update the policy.

    This function will only be called once to create a TF graph which
    will be run repeatedly during training at the learner.

    All the arguments are tf placeholders (or nested structures of placeholders).

    ([step_type, rew, obs, discount], prev_state) -> step_output,
    Args:
      step_outputs: [T, B, ... ] of StepOutput structures.
      prev_states: [T + 1, B, ...] of agent prev states.
      step_types: [T + 1, B, ...] of stepTypes
      rewards: [T + 1, B, ...] of reward values
      observations: [T + 1, B, ...] of env observations.
      discounts: [T + 1, B] of discount values at each step.
    """
    config = self.config
    with tf.variable_scope(self._name):
      # flatten graph features for policy network
      # get logits
      # logits -> [T* B, ...]
      target_logits, _, _ = self._model.get_logits_and_next_state(
          *nest.map_structure(merge_first_two_dims, [
              step_types[:-1],
              rewards[:-1],
              nest.map_structure(lambda k: k[:-1], observations),
              prev_states[:-1],
          ]))

      # get value.
      # [(T+1)* B]
      values = self._model.get_value(
          *nest.map_structure(merge_first_two_dims, [
              step_types,
              rewards,
              observations,
              prev_states,
          ]))

      t_dim = infer_shape(step_types)[0] - 1
      bs_dim = infer_shape(step_types)[1]
      values = tf.reshape(values, [t_dim + 1, bs_dim])

      actions = step_outputs.action  # [T, B]
      behavior_logits = step_outputs.logits  # [T, B]
      # [T, B]
      target_logits = tf.reshape(target_logits, infer_shape(behavior_logits))

      self.loss = VTraceLoss(step_types, actions, rewards, discounts,
                             behavior_logits, target_logits, values,
                             config.discount_factor,
                             self._get_entropy_regularization_constant(),
                             **config.loss)

      valid_mask = ~tf.equal(step_types[1:], StepType.FIRST)
      n_valid_steps = tf.cast(tf.reduce_sum(tf.cast(valid_mask, tf.int32)),
                              tf.float32)
      opt_vals = self._optimize(self.loss.loss)

      def f(x):
        """Computes the valid mean stat."""
        return tf.reduce_sum(tf.boolean_mask(x, valid_mask)) / n_valid_steps

      # TODO: Add histogram summaries
      # https://github.com/google-research/batch-ppo/blob/master/agents/algorithms/ppo/utility.py
      self._logged_values = {
          # entropy
          'entropy/target_policy_entropy':
          f(compute_entropy(target_logits)),
          'entropy/behavior_policy_entropy':
          f(compute_entropy(behavior_logits)),
          'entropy/is_ratio':
          f(
              tf.exp(-tf.nn.sparse_softmax_cross_entropy_with_logits(
                  labels=actions, logits=target_logits) +
                     tf.nn.sparse_softmax_cross_entropy_with_logits(
                         labels=actions, logits=behavior_logits))),

          # rewards
          'reward/avg_reward':
          f(rewards[1:]),
          **opt_vals,
          **self._extract_logged_values(
              tf.nest.map_structure(lambda k: k[:-1], observations), f),
          **self.loss.logged_values
      }

  def update(self, sess, feed_dict, profile_kwargs):
    """profile_kwargs pass to sess.run for profiling purposes."""
    _, vals = sess.run([self._train_op, self._logged_values],
                       feed_dict=feed_dict,
                       **profile_kwargs)
    return vals
