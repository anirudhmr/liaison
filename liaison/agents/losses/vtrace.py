import tensorflow as tf
from liaison.agents.losses import vtrace_ops
from liaison.agents.utils import *
from liaison.env import StepType
from liaison.utils import ConfigDict


class Loss:

  def __init__(self,
               step_types,
               actions,
               rewards,
               discounts,
               behavior_logits,
               target_logits,
               values,
               discount_factor,
               entropy_coeff,
               vf_loss_coeff,
               bootstrap_value,
               clip_rho_threshold=1.0,
               clip_pg_rho_threshold=1.0,
               **kwargs):
    """Policy gradient loss with vtrace importance weighting.
        VTraceLoss takes tensors of shape [T, B, ...], where `B` is the
        batch_size. The reason we need to know `B` is for V-trace to properly
        handle episode cut boundaries.
        Args:
            step_types: [T + 1, B, ...] of stepTypes
            actions: An int|float32 tensor of shape [T, B, ACTION_SPACE].
            rewards: [T + 1, B, ...] of reward values
            discounts: [T + 1, B] of discount values at each step.
            behaviour_logits: [T, B] of behavior policy logits
            target_logits: [T, B] of target policy logits
            values: [T + 1, B] A float32 tensor of shape.
            entropy_coeff: Coefficient to use for entropy regularization.
    """
    t_dim = infer_shape(step_types)[0] - 1
    bs_dim = infer_shape(step_types)[1]
    # [T, B]
    values = values[:-1]

    # Compute vtrace on the CPU for better perf.
    # with tf.device("/cpu:0"):
    vtrace_returns = vtrace_ops.from_logits(
        behaviour_policy_logits=behavior_logits,
        target_policy_logits=target_logits,
        actions=actions,
        discounts=discounts[1:] * discount_factor,
        rewards=rewards[1:],
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=tf.cast(clip_rho_threshold, tf.float32),
        clip_pg_rho_threshold=tf.cast(clip_pg_rho_threshold, tf.float32))

    # Ignore the timesteps that caused a reset to happen
    # [T, B]
    valid_mask = ~tf.equal(step_types[1:], StepType.FIRST)
    n_valid_steps = tf.cast(tf.reduce_sum(tf.cast(valid_mask, tf.int32)), tf.float32)

    actions_logp = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions,
                                                                   logits=target_logits)

    # The policy gradients loss
    pi_loss = -tf.reduce_sum(
        tf.boolean_mask(actions_logp * vtrace_returns.pg_advantages, valid_mask))

    # The baseline loss
    delta = tf.boolean_mask(values - vtrace_returns.vs, valid_mask)
    vf_loss = 0.5 * tf.reduce_sum(tf.square(delta))

    # The entropy for valid actions
    entropy = tf.reduce_sum(tf.boolean_mask(compute_entropy(target_logits), valid_mask))

    # The summed weighted loss
    total_loss = (pi_loss + vf_loss * vf_loss_coeff - entropy * entropy_coeff)
    # scale it for per step units.
    total_loss /= n_valid_steps

    def f(x):
      """Computes the valid mean stat."""
      return tf.reduce_sum(tf.boolean_mask(x, valid_mask)) / n_valid_steps

    self._logged_values = {
        # loss
        'loss/entropy_loss': -entropy * entropy_coeff / n_valid_steps,
        'loss/pg_loss': pi_loss / n_valid_steps,
        'loss/vf_loss': vf_loss * vf_loss_coeff / n_valid_steps,
        'loss/total_loss': total_loss,
        # rewards
        'reward/advantage': f(vtrace_returns.pg_advantages),
        'reward/vtrace_returns': f(vtrace_returns.vs),
    }
    self.total_loss = total_loss

  @property
  def loss(self):
    return self.total_loss

  @property
  def logged_values(self):
    return self._logged_values


class MultiActionLoss(Loss):

  def __init__(self,
               step_types,
               actions,
               rewards,
               discounts,
               behavior_logits,
               target_logits,
               values,
               discount_factor,
               entropy_coeff,
               vf_loss_coeff,
               bootstrap_value,
               clip_rho_threshold=1.0,
               clip_pg_rho_threshold=1.0,
               action_mask=None,
               **kwargs):
    """Policy gradient loss with vtrace importance weighting.
        VTraceLoss takes tensors of shape [T, B, ...], where `B` is the
        batch_size. The reason we need to know `B` is for V-trace to properly
        handle episode cut boundaries.
        Args:
            step_types: [T + 1, B, ...] of stepTypes
            actions: An int|float32 tensor of shape [T, B, N, ACTION_SPACE].
            rewards: [T + 1, B, ...] of reward values
            discounts: [T + 1, B] of discount values at each step.
            behaviour_logits: [T, B, N, d] of behavior policy logits
            target_logits: [T, B, N, d] of target policy logits
            values: [T + 1, B] A float32 tensor of shape.
            entropy_coeff: Coefficient to use for entropy regularization.
    """
    t_dim = infer_shape(step_types)[0] - 1
    bs_dim = infer_shape(step_types)[1]
    # [T, B]
    values = values[:-1]

    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    if action_mask is not None:
      action_mask = tf.cast(action_mask, tf.float32)

    with tf.name_scope('vtrace_from_logits',
                       values=[
                           behavior_logits, target_logits, actions, discounts, rewards, values,
                           bootstrap_value
                       ]):
      target_action_log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=target_logits, labels=actions)
      # sum up logprobs of the sub-actions at each timestep.
      if action_mask is not None:
        target_action_log_probs = tf.reduce_sum(target_action_log_probs * action_mask, -1)
      else:
        target_action_log_probs = tf.reduce_sum(target_action_log_probs, -1)

      behaviour_action_log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=behavior_logits, labels=actions)

      if action_mask is not None:
        behaviour_action_log_probs = tf.reduce_sum(behaviour_action_log_probs * action_mask, -1)
      else:
        behaviour_action_log_probs = tf.reduce_sum(behaviour_action_log_probs, -1)

      log_rhos = target_action_log_probs - behaviour_action_log_probs
      vtrace_returns = vtrace_ops.from_importance_weights(
          log_rhos=log_rhos,
          discounts=discounts[1:] * discount_factor,
          rewards=rewards[1:],
          values=values,
          bootstrap_value=bootstrap_value,
          clip_rho_threshold=tf.cast(clip_rho_threshold, tf.float32),
          clip_pg_rho_threshold=tf.cast(clip_pg_rho_threshold, tf.float32))

      vtrace_returns = vtrace_ops.VTraceFromLogitsReturns(
          log_rhos=log_rhos,
          behaviour_action_log_probs=behaviour_action_log_probs,
          target_action_log_probs=target_action_log_probs,
          **vtrace_returns._asdict())

    # Ignore the timesteps that caused a reset to happen
    # [T, B]
    valid_mask = ~tf.equal(step_types[1:], StepType.FIRST)
    n_valid_steps = tf.cast(tf.reduce_sum(tf.cast(valid_mask, tf.int32)), tf.float32)

    # The policy gradients loss
    pi_loss = -tf.reduce_sum(
        tf.boolean_mask(target_action_log_probs * vtrace_returns.pg_advantages, valid_mask))

    # The baseline loss
    delta = tf.boolean_mask(values - vtrace_returns.vs, valid_mask)
    vf_loss = 0.5 * tf.reduce_sum(tf.square(delta))

    if action_mask is not None:
      # The entropy for valid actions
      ent = action_mask * compute_entropy(target_logits)
      entropy = tf.reduce_sum(tf.boolean_mask(tf.reduce_mean(ent, -1), valid_mask))
    else:
      entropy = tf.reduce_sum(
          tf.boolean_mask(tf.reduce_mean(compute_entropy(target_logits), -1), valid_mask))

    # The summed weighted loss
    total_loss = (pi_loss + vf_loss * vf_loss_coeff - entropy * entropy_coeff)
    # scale it for per step units.
    total_loss /= n_valid_steps

    def f(x):
      """Computes the valid mean stat."""
      return tf.reduce_sum(tf.boolean_mask(x, valid_mask)) / n_valid_steps

    self._logged_values = {
        # loss
        'loss/entropy_loss': -entropy * entropy_coeff / n_valid_steps,
        'loss/pg_loss': pi_loss / n_valid_steps,
        'loss/vf_loss': vf_loss * vf_loss_coeff / n_valid_steps,
        'loss/total_loss': total_loss,
        # rewards
        'reward/advantage': f(vtrace_returns.pg_advantages),
        'reward/vtrace_returns': f(vtrace_returns.vs),
    }
    self.total_loss = total_loss
