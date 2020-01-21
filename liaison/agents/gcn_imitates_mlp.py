import graph_nets as gn
import liaison.utils as U
import numpy as np
import tensorflow as tf
import tree as nest
from liaison.agents import BaseAgent, StepOutput
from liaison.agents.losses.vtrace import Loss as VTraceLoss
from liaison.agents.utils import *
from liaison.env import StepType
from liaison.utils import ConfigDict


class Agent(BaseAgent):

  def __init__(self,
               name,
               action_spec,
               seed,
               mlp_model,
               model,
               evaluation_mode=False,
               **kwargs):

    self.set_seed(seed)
    self.config = ConfigDict(evaluation_mode=evaluation_mode, **kwargs)
    self._name = name
    self._action_spec = action_spec
    mlp_model.update(action_spec=action_spec)
    self._load_models(name, model, mlp_model)

  def _load_models(self, name, model, mlp_model):
    # Loads a network based on the provided config.
    model_class = U.import_obj('Model', model.class_path)
    mlp_model_class = U.import_obj('Model', mlp_model.class_path)
    del model['class_path'], mlp_model['class_path']
    with tf.variable_scope(name):
      self._mlp_model = mlp_model_class(seed=self.seed, **mlp_model)
      with tf.variable_scope('gcn'):
        self._model = model_class(seed=self.seed, **model)

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
      if self.config.evaluation_mode:
        # evaluate gcn agent at the evaluators.
        # flatten graph features for the policy network
        # convert dict to graphstuple
        graph_features = gn.graphs.GraphsTuple(**obs['graph_features'])
        obs['graph_features'] = flatten_graphs(graph_features)

        logits, _ = self._model.get_logits(
            self._model.compute_graph_embeddings(obs), obs['node_mask'])

        action = sample_from_logits(logits, self.seed)
        return StepOutput(action, logits,
                          self._model.dummy_state(infer_shape(step_type)[0]))
      else:
        # use mlp to imitate during training mode in actors.
        logits, next_state, _ = self._mlp_model.get_logits_and_next_state(
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
    behavior_logits = step_outputs.logits  # [T, B, N]
    actions = step_outputs.action  # [T, B]

    with tf.variable_scope(self._name):
      # call to ensure mlp model is built and variables are created at the learenr.
      self._mlp_model.get_logits_and_next_state(
          *nest.map_structure(merge_first_two_dims, [
              step_types[:-1],
              rewards[:-1],
              nest.map_structure(lambda k: k[:-1], observations),
              prev_states[:-1],
          ]))

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
        # target_logits -> [(T + 1)* B, ...]
        target_logits, logits_logged_vals = self._model.get_logits(
            graph_embeddings, flattened_observations['node_mask'])
        # reshape to [T + 1, B ...]
        target_logits = tf.reshape(target_logits, [t_dim + 1, bs_dim, -1])

        # reshape to [T, B ...]
        target_logits = target_logits[:-1]

      labels = tf.nn.softmax(behavior_logits)
      padding = np.zeros((3, 2))
      padding[-1, 1] = infer_shape(target_logits)[-1] - infer_shape(labels)[-1]
      labels = tf.pad(labels, padding, constant_values=0.)
      loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                     logits=target_logits)
      loss = tf.reduce_mean(loss)

      with tf.variable_scope('optimize'):
        opt_vals = self._optimize(loss)

      with tf.variable_scope('logged_vals'):
        valid_mask = ~tf.equal(step_types[1:], StepType.FIRST)
        n_valid_steps = tf.cast(tf.reduce_sum(tf.cast(valid_mask, tf.int32)),
                                tf.float32)

        def f(x):
          """Computes the valid mean stat."""
          return tf.reduce_sum(tf.boolean_mask(x, valid_mask)) / n_valid_steps

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
            'loss/cross_entropy_loss':
            loss,
            **opt_vals,
            **logits_logged_vals,
            **self._extract_logged_values(
                nest.map_structure(lambda k: k[:-1], observations), f),
        }

  def update(self, sess, feed_dict, profile_kwargs):
    """profile_kwargs pass to sess.run for profiling purposes."""
    _, vals = sess.run([self._train_op, self._logged_values],
                       feed_dict=feed_dict,
                       **profile_kwargs)
    return vals
