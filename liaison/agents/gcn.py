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
import graph_nets as gn


class Agent(BaseAgent):

  def __init__(self, name, action_spec, seed, model=None, **kwargs):

    self.set_seed(seed)
    self.config = ConfigDict(**kwargs)
    self._name = name
    self._action_spec = action_spec
    self._load_model(name, action_spec=action_spec, **(model or {}))

  def initial_state(self, bs):
    return self._model.get_initial_state(bs)

  def _flatten_graphs(self, graph_features):
    """
      Flatten graphs. Remove padding.
      Args:
        graph_features: gn.graphs.GraphsTuple.
        B is batch size
        M_N is the max # of node (graphs with < M_N nodes use padding)
        M_E max # of edges.
        graph_features.nodes    => [B, M_N, ...]
        graph_features.edges    => [B, M_E, ...]
        graph_features.senders  => [B, M_E]
        graph_features.receivers=> [B, M_E]
        graph_features.n_node   => [B]
        graph_features.n_edge   => [B]
        graph_features.globals  => [B, ...]
      Returns:
        graph_features: gn.graphs.GraphsTuple
        Let S_N = sum(graph_features.n_node)
        Let S_E = sum(graph_features.n_edge)
        graph_features.nodes    => [S_N, ...]
        graph_features.edges    => [S_E, ...]
        graph_features.senders  => [S_E]
        graph_features.receivers=> [S_E]
        graph_features.n_node   => [B]
        graph_features.n_edge   => [B]
        graph_features.globals  => [B, ...]
    """
    node_indices = gn.utils_tf.sparse_to_dense_indices(graph_features.n_node)
    edge_indices = gn.utils_tf.sparse_to_dense_indices(graph_features.n_edge)
    graph_features = graph_features.replace(
        nodes=tf.gather_nd(params=graph_features.nodes, indices=node_indices),
        edges=tf.gather_nd(params=graph_features.edges, indices=edge_indices),
        senders=tf.gather_nd(params=graph_features.senders,
                             indices=edge_indices),
        receivers=tf.gather_nd(params=graph_features.receivers,
                               indices=edge_indices))
    return gn.utils_tf.stop_gradient(graph_features)

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

    self._validate_observations(obs)
    with tf.variable_scope(self._name):
      # flatten graph features for the policy network
      # convert dict to graphstuple
      graph_features = gn.graphs.GraphsTuple(**obs['graph_features'])
      obs['graph_features'] = self._flatten_graphs(graph_features)

      logits, next_state, _ = self._model.get_logits_and_next_state(
          step_type, reward, obs, prev_state)

      action = sample_from_logits(logits, self.seed)
      return StepOutput(action, logits, next_state)

  def _validate_observations(self, obs):
    if 'graph_features' not in obs:
      raise Exception('graph_features not found in observation.')
    elif 'node_mask' not in obs:
      raise Exception('node_mask not found in observation.')

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

    self._validate_observations(observations)
    config = self.config
    with tf.variable_scope(self._name):
      # flatten graph features for policy network
      with tf.variable_scope('flatten_graphs_for_logits'):
        # time dimension T + 1 => T
        observations_minus_1 = nest.map_structure(lambda k: k[:-1],
                                                  observations)
        # merge time and batch dimensions
        observations_minus_1 = nest.map_structure(merge_first_two_dims,
                                                  observations_minus_1)
        graph_features = gn.graphs.GraphsTuple(
            **observations_minus_1['graph_features'])
        # flatten by merging the batch and node, edge dimensions
        graph_features = self._flatten_graphs(graph_features)
        observations_minus_1['graph_features'] = graph_features

      with tf.variable_scope('target_logits'):
        # get logits
        # logits -> [T* B, ...]
        target_logits, _, logits_logged_vals = self._model.get_logits_and_next_state(
            *nest.map_structure(merge_first_two_dims,
                                [step_types[:-1], rewards[:-1]]),
            observations_minus_1,
            *nest.map_structure(merge_first_two_dims, [prev_states[:-1]]))

      with tf.variable_scope('flatten_graphs_for_value_func'):
        # flatten graphs for value network
        # merge time and batch dimensions
        observations = nest.map_structure(merge_first_two_dims, observations)
        graph_features = gn.graphs.GraphsTuple(
            **observations['graph_features'])
        # flatten by merging the batch and node, edge dimensions
        graph_features = self._flatten_graphs(graph_features)
        observations['graph_features'] = graph_features

      with tf.variable_scope('value'):
        # get value.
        # [(T+1)* B]
        values = self._model.get_value(
            *nest.map_structure(merge_first_two_dims, [step_types, rewards]),
            observations,
            *nest.map_structure(merge_first_two_dims, [prev_states]))

      with tf.variable_scope('loss'):
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

      with tf.variable_scope('optimize'):
        opt_vals = self._optimize(self.loss.loss)
      with tf.variable_scope('logged_vals'):
        valid_mask = ~tf.equal(step_types[1:], StepType.FIRST)
        n_valid_steps = tf.cast(tf.reduce_sum(tf.cast(valid_mask, tf.int32)),
                                tf.float32)

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
            **logits_logged_vals,
            **self.loss.logged_values
        }

  def update(self, sess, feed_dict, profile_kwargs):
    """profile_kwargs pass to sess.run for profiling purposes."""
    _, vals = sess.run([self._train_op, self._logged_values],
                       feed_dict=feed_dict,
                       **profile_kwargs)
    return vals
