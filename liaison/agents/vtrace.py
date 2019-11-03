import numpy as np
import tensorflow as tf
from absl import logging
from liaison.agents import BaseAgent, StepOutput, utils, vtrace_ops
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
    self._load_model(name, **(model or {}))
    self._obs_ph = None

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
      logits, next_state = self._model.get_logits_and_next_state(
          step_type, reward, obs, prev_state)
      if 'mask' in obs:
        mask = tf.reshape(mask, tf.shape(logits))
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
    self._obs_ph = observations
    with tf.variable_scope(self._name):
      # logits -> [T* B, ...]
      target_logits, _ = self._model.get_logits_and_next_state(
          *nest.map_structure(merge_first_two_dims, [
              step_types[:-1],
              rewards[:-1],
              nest.map_structure(lambda k: k[:-1], observations),
              prev_states[:-1],
          ]))

      t_dim = infer_shape(step_types)[0] - 1
      bs_dim = infer_shape(step_types)[1]

      actions = step_outputs.action  # [T, B]
      behavior_logits = step_outputs.logits  # [T, B]
      # [T, B]
      target_logits = tf.reshape(target_logits, infer_shape(behavior_logits))

      # [(T+1)* B]
      values = self._model.get_value(
          *nest.map_structure(merge_first_two_dims, [
              step_types,
              rewards,
              observations,
              prev_states,
          ]))
      values = tf.reshape(values, [t_dim + 1, bs_dim])
      # [B]
      bootstrap_value = tf.reshape(values[-1], [bs_dim])
      # [T, B]
      values = values[:-1]

      # Compute vtrace on the CPU for better perf.
      with tf.device("/cpu:0"):
        vtrace_returns = vtrace_ops.from_logits(
            behaviour_policy_logits=behavior_logits,
            target_policy_logits=target_logits,
            actions=actions,
            discounts=discounts[1:] * config.discount_factor,
            rewards=rewards[1:],
            values=values,
            bootstrap_value=bootstrap_value,
            clip_rho_threshold=tf.cast(config.clip_rho_threshold, tf.float32),
            clip_pg_rho_threshold=tf.cast(config.clip_pg_rho_threshold,
                                          tf.float32))

      # Ignore the timesteps that caused a reset to happen
      # [T, B]
      valid_mask = ~tf.equal(step_types[1:], StepType.FIRST)
      n_valid_steps = tf.cast(tf.reduce_sum(tf.cast(valid_mask, tf.int32)),
                              tf.float32)

      actions_logp = -tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=actions, logits=target_logits)

      # The policy gradients loss
      pi_loss = -tf.reduce_sum(
          tf.boolean_mask(actions_logp * vtrace_returns.pg_advantages,
                          valid_mask))

      # The baseline loss
      delta = tf.boolean_mask(values - vtrace_returns.vs, valid_mask)
      vf_loss = 0.5 * tf.reduce_sum(tf.square(delta))

      # The entropy for valid actions
      entropy = tf.reduce_sum(
          tf.boolean_mask(compute_entropy(target_logits), valid_mask))

      entropy_coeff = self._get_entropy_regularization_constant()
      # The summed weighted loss
      total_loss = (pi_loss + vf_loss * config.loss.vf_loss_coeff -
                    entropy * entropy_coeff)
      # scale it for per step units.
      total_loss /= n_valid_steps

      self.global_step = tf.train.get_or_create_global_step()
      lr = self._lr_schedule()
      optimizer = self._init_optimizer(lr)

      # get clipped gradients
      grads, variables = zip(*optimizer.compute_gradients(total_loss))
      cli_grads, global_norm = tf.clip_by_global_norm(grads, config.grad_clip)
      clipped_grads_and_vars = list(zip(cli_grads, variables))
      self._train_op = optimizer.apply_gradients(clipped_grads_and_vars,
                                                 global_step=self.global_step)

      def f(x):
        """Computes the valid mean stat."""
        return tf.reduce_sum(tf.boolean_mask(x, valid_mask)) / n_valid_steps

      # TODO: Add histogram summaries
      # https://github.com/google-research/batch-ppo/blob/master/agents/algorithms/ppo/utility.py
      # Also add episode statistics from environment to this dict.
      self._logged_values = {
          # entropy
          'entropy/target_policy_entropy':
          f(compute_entropy(target_logits)),
          'entropy/behavior_policy_entropy':
          f(compute_entropy(behavior_logits)),
          'entropy/is_ratio':
          f(
              tf.exp(actions_logp +
                     tf.nn.sparse_softmax_cross_entropy_with_logits(
                         labels=actions, logits=behavior_logits))),
          # loss
          'loss/entropy_loss':
          -entropy * entropy_coeff / n_valid_steps,
          'loss/pg_loss':
          pi_loss / n_valid_steps,
          'loss/vf_loss':
          vf_loss * config.loss.vf_loss_coeff / n_valid_steps,
          'loss/total_loss':
          total_loss,
          # optimization related
          'opt/pre_clipped_grad_norm':
          global_norm,
          'opt/clipped_grad_norm':
          tf.linalg.global_norm(cli_grads),
          'opt/lr':
          lr,
          'opt/weight_norm':
          tf.linalg.global_norm(variables),
          # rewards
          'reward/advantage':
          f(vtrace_returns.pg_advantages),
          'reward/avg_reward':
          f(rewards[1:]),
          'reward/vtrace_returns':
          f(vtrace_returns.vs),
      }

  def _init_optimizer(self, lr_op):
    return tf.train.AdamOptimizer(lr_op)

  def update(self, sess, feed_dict):
    _, vals = sess.run([self._train_op, self._logged_values],
                       feed_dict=feed_dict)
    return vals
