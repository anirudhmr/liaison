import graph_nets as gn
import tensorflow as tf
import tree as nest
from liaison.agents import BaseAgent, StepOutput
from liaison.agents.losses.vtrace import Loss as VTraceLoss
from liaison.agents.utils import *
from liaison.env import StepType
from liaison.utils import ConfigDict


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

    self._validate_observations(obs)
    with tf.variable_scope(self._name):
      # flatten graph features for the policy network
      # convert dict to graphstuple
      graph_features = gn.graphs.GraphsTuple(**obs['graph_features'])
      obs['graph_features'] = flatten_graphs(graph_features)

      logits, _ = self._model.get_logits(
          self._model.compute_graph_embeddings(obs), obs['node_mask'])

      action = sample_from_logits(logits, self.seed)
      return StepOutput(action, logits,
                        self._model.dummy_state(infer_shape(step_type)[0]))

  def _validate_observations(self, obs):
    if 'graph_features' not in obs:
      raise Exception('graph_features not found in observation.')

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
    behavior_logits = step_outputs.logits  # [T, B]
    actions = step_outputs.action  # [T, B]

    with tf.variable_scope(self._name):
      t_dim = infer_shape(step_types)[0] - 1
      bs_dim = infer_shape(step_types)[1]

      # flatten graph features for graph embeddings.
      with tf.variable_scope('flatten_graphs'):
        # merge time and batch dimensions
        flattened_observations = nest.map_structure(merge_first_two_dims,
                                                    observations)
        # flatten by merging the batch and node, edge dimensions
        flattened_observations['graph_features'] = flatten_graphs(
            gn.graphs.GraphsTuple(**flattened_observations['graph_features']))

      with tf.variable_scope('graph_embeddings'):
        graph_embeddings = self._model.compute_graph_embeddings(
            flattened_observations)

      with tf.variable_scope('target_logits'):
        # get logits
        # target_logits -> [(T + 1)* B, ...]
        target_logits, logits_logged_vals = self._model.get_logits(
            graph_embeddings, flattened_observations['node_mask'])
        # reshape to [T + 1, B ...]
        target_logits = tf.reshape(target_logits, [t_dim + 1, bs_dim] +
                                   infer_shape(behavior_logits)[2:])
        # reshape to [T, B ...]
        target_logits = target_logits[:-1]

      with tf.variable_scope('value'):
        # get value.
        # [(T+1)* B]
        values = self._model.get_value(graph_embeddings)

      if config.loss.al_coeff.init_val > 0:
        with tf.variable_scope('auxiliary_supervised_loss'):
          auxiliary_loss = self._model.get_auxiliary_loss(
              graph_embeddings, flattened_observations)

      with tf.variable_scope('loss'):
        values = tf.reshape(values, [t_dim + 1, bs_dim])

        if 'bootstrap_value' in observations:
          bootstrap_value = observations['bootstrap_value'][-1]
        else:
          bootstrap_value = values[-1]

        self.loss = VTraceLoss(step_types,
                               actions,
                               rewards,
                               discounts,
                               behavior_logits,
                               target_logits,
                               values,
                               config.discount_factor,
                               self._get_entropy_regularization_constant(),
                               bootstrap_value=bootstrap_value,
                               **config.loss)
        loss = self.loss.loss
        if config.loss.al_coeff.init_val > 0:
          loss += auxiliary_loss * get_decay_ops(**config.loss.al_coeff)
          self.loss.logged_values.update({
              'loss/auxiliary_loss': auxiliary_loss,
          })
        self.loss.logged_values.update({
            'loss/total_loss': loss,
        })

      with tf.variable_scope('optimize'):
        opt_vals = self._optimize(loss)

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
            'entropy/uniform_random_entropy':
            f(
                compute_entropy(
                    tf.cast(tf.greater(target_logits, -1e8), tf.float32))),
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
            **self._extract_logged_values(
                nest.map_structure(lambda k: k[:-1], observations), f),
            **self.loss.logged_values
        }

  def update(self, sess, feed_dict, profile_kwargs):
    """profile_kwargs pass to sess.run for profiling purposes."""
    _, vals = sess.run([self._train_op, self._logged_values],
                       feed_dict=feed_dict,
                       **profile_kwargs)
    return vals
