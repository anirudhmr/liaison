import pdb

import graph_nets as gn
import tensorflow as tf
import tree as nest
from liaison.agents import BaseAgent, StepOutput
from liaison.agents.losses.vtrace import \
    MultiActionLoss as VTraceMultiActionLoss
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
    self._global_step = tf.train.get_or_create_global_step()
    self._total_steps = tf.Variable(0,
                                    trainable=False,
                                    collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                    name='total_steps')

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
      pack_as_structure = dict(**obs['graph_features'])
      obs['graph_features'] = self._process_graph_features(obs['graph_features'])
      ge = self._model.compute_graph_embeddings(obs)
      logitss, actions = self._model.get_actions(ge, obs)
      # pack by padding to the max nodes.
      packed_ge = self._model.pack_graph_embeddings(pack_as_structure, ge)
      return StepOutput(actions, logitss, self._model.dummy_state(infer_shape(step_type)[0]),
                        dict(**packed_ge._asdict()))

  def _validate_observations(self, obs):
    if 'graph_features' not in obs:
      raise Exception('graph_features not found in observation.')

  def _process_graph_features(self, graph_features):
    if set(graph_features.keys()) == set(gn.graphs.GraphsTuple._fields):
      return flatten_graphs(gn.graphs.GraphsTuple(**graph_features))
    elif set(graph_features.keys()) == set(gn.graphs.BipartiteGraphsTuple._fields):
      return flatten_bipartite_graphs(gn.graphs.BipartiteGraphsTuple(**graph_features))
    else:
      raise Exception('Unknown graph features provided.')

  def build_update_ops(self, step_outputs, prev_states, step_types, rewards, observations,
                       discounts):
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
    behavior_logits = step_outputs.logits  # [T, B, T2, T1]
    actions = step_outputs.action  # [T, B, T2]
    packed_ge = dict(**step_outputs.graph_embeddings)  # [T, B, ...]

    with tf.variable_scope(self._name):
      t_dim = infer_shape(step_types)[0] - 1
      bs_dim = infer_shape(step_types)[1]

      # flatten graph features for graph embeddings.
      with tf.variable_scope('flatten_graphs'):
        # merge time and batch dimensions
        flattened_observations = nest.map_structure(merge_first_two_dims, observations)
        # flatten by merging the batch and node, edge dimensions
        flattened_observations['graph_features'] = self._process_graph_features(
            flattened_observations['graph_features'])

      with tf.variable_scope('unpack_graph_embeddings'):
        packed_ge = nest.map_structure(merge_first_two_dims, packed_ge)
        packed_ge['edges'] = None
        packed_ge = self._process_graph_features(packed_ge)
        packed_ge = packed_ge.replace(edges=tf.fill(tf.expand_dims(bs_dim, 0), 0.))

      def f(graph):
        graph_dict = graph._asdict()
        for k in ['n_left_nodes', 'n_right_nodes', 'n_node', 'globals']:
          if k in graph_dict:
            graph = graph.replace(**{k: graph_dict[k][:t_dim * bs_dim]})
        return graph

      # compute graph_embeddings or load from cache
      graph_embeddings = tf.cond(
          self._global_step < config.freeze_graphnet_weights_step,
          lambda: f(self._model.compute_graph_embeddings(flattened_observations)),
          lambda: packed_ge,
      )
      # get logits
      # target_logits -> [T * B, ...]
      target_logits, _, self._logged_features = self._model.get_actions(graph_embeddings,
                                                                        flattened_observations,
                                                                        step_types,
                                                                        sampled_actions=tf.reshape(
                                                                            actions,
                                                                            [t_dim * bs_dim, -1]),
                                                                        log_features=True)
      assert infer_shape(target_logits)[0] == bs_dim * t_dim
      target_logits = tf.reshape(target_logits, [t_dim, bs_dim] + infer_shape(behavior_logits)[2:])

      with tf.variable_scope('value'):
        # get value.
        # [(T+1)* B]
        values = self._model.get_value(graph_embeddings, flattened_observations)

      if config.loss.al_coeff.init_val > 0:
        raise Exception('Not supported just yet!')

      with tf.variable_scope('loss'):
        # bug in previous versions.
        # Note that the bootstrap value has to be according to the target policy.
        # Hence it cannot be computed from the actor's policy.
        # if 'bootstrap_value' in observations:
        #   bootstrap_value = observations['bootstrap_value'][-1]
        # else:
        last_obs = nest.map_structure(lambda v: v[-1], observations)
        last_obs['graph_features'] = self._process_graph_features(last_obs['graph_features'])
        ge = self._model.compute_graph_embeddings(last_obs)
        ge = tf.cond(self._global_step < config.freeze_graphnet_weights_step, lambda: ge,
                     lambda: ge.map(tf.stop_gradient))
        bootstrap_value = self._model.get_value(ge, last_obs)
        bootstrap_value = tf.stop_gradient(bootstrap_value)
        values = tf.concat([values, bootstrap_value], axis=0)
        values = tf.reshape(values, [t_dim + 1, bs_dim])

        # Ex: convert [1, 2] -> [[1,0], [1, 1]]
        actions_flattened = tf.reshape(actions, [t_dim * bs_dim, infer_shape(actions)[-1]])
        indices = gn.utils_tf.sparse_to_dense_indices(
            merge_first_two_dims(observations['n_actions'][:-1]))
        action_mask = tf.scatter_nd(indices, tf.ones(infer_shape(indices)[:1], dtype=tf.bool),
                                    infer_shape(actions_flattened))
        action_mask = tf.reshape(action_mask, [t_dim, bs_dim, -1])
        self.loss = VTraceMultiActionLoss(step_types,
                                          actions,
                                          rewards,
                                          discounts,
                                          behavior_logits,
                                          target_logits,
                                          values,
                                          config.discount_factor,
                                          self._get_entropy_regularization_constant(),
                                          bootstrap_value=bootstrap_value,
                                          action_mask=action_mask,
                                          **config.loss)
        loss = self.loss.loss
        if config.loss.al_coeff.init_val > 0:
          raise Exception('Not supported just yet!')

        self.loss.logged_values.update({
            'loss/total_loss': loss,
        })

      with tf.variable_scope('optimize'):
        with tf.control_dependencies([tf.assign_add(self._total_steps, t_dim * bs_dim)]):
          opt_vals = self._optimize(loss)

      with tf.variable_scope('logged_vals'):
        valid_mask = ~tf.equal(step_types[1:], StepType.FIRST)

        for statistic in ('mean', 'std'):

          def f(x):
            """Computes the valid mean stat."""
            if statistic == 'mean':
              return tf.reduce_mean(tf.boolean_mask(x, valid_mask))
            else:
              return tf.math.reduce_std(tf.boolean_mask(x, valid_mask))

          def f2(x):
            y = tf.cast(action_mask, x.dtype)
            return tf.reduce_sum(y * x, -1) / tf.reduce_sum(y, -1)

          logged_values = {
              # entropy
              'entropy/uniform_random_entropy':
              f(f2(compute_entropy(tf.cast(tf.greater(target_logits, -1e8), tf.float32)))),
              'entropy/target_policy_entropy':
              f(f2(compute_entropy(target_logits))),
              'entropy/behavior_policy_entropy':
              f(f2(compute_entropy(behavior_logits))),
              'entropy/is_ratio':
              f(
                  f2(
                      tf.exp(-tf.nn.sparse_softmax_cross_entropy_with_logits(
                          labels=actions, logits=target_logits) +
                             tf.nn.sparse_softmax_cross_entropy_with_logits(
                                 labels=actions, logits=behavior_logits)))),
              # rewards
              'reward/reward':
              f(rewards[1:]),
              **self._extract_logged_values(nest.map_structure(lambda k: k[:-1], observations), f),
          }
          if statistic == 'mean':
            self._logged_values_mean = {
                **logged_values,
                'steps/total_steps': tf.reduce_mean(self._total_steps),
                **self.loss.logged_values,
                **opt_vals,
            }
          else:
            self._logged_values_std = {**logged_values, **self.loss.logged_values_std}

  def _log_features(self, features, step):
    # feature_dict -> collected Feature
    # logs t-SNE visualization of features to files
    if 'vis_loggers' in self.config:
      self._model.log_features(features, step, self.config.vis_loggers)

  def update(self, sess, feed_dict, profile_kwargs):
    """profile_kwargs pass to sess.run for profiling purposes."""
    ops = []
    i = sess.run(self._global_step)
    log_features = False
    if self.config.log_features_every > 0:
      if i % self.config.log_features_every == 0:
        ops += [self._logged_features]
        log_features = True

    mean_vals, var_vals, _, *l = sess.run(
        [self._logged_values_mean, self._logged_values_std, self._train_op] + ops,
        feed_dict=feed_dict,
        **profile_kwargs)

    if log_features:
      self._log_features(l[0], i)
    return mean_vals, var_vals
