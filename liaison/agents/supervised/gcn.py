import graph_nets as gn
import tensorflow as tf
from liaison.agents.utils import flatten_graphs
from liaison.utils import ConfigDict

from .base import Agent as BaseAgent


class Agent(BaseAgent):

  def __init__(self, name, action_spec, seed, model=None, **kwargs):

    self.set_seed(seed)
    self.config = ConfigDict(**kwargs)
    self._name = name
    self._action_spec = action_spec
    self._load_model(name, action_spec=action_spec, **(model or {}))
    self._logged_values = None

  def _validate_observations(self, obs):
    for k in ['graph_features', 'node_mask']:
      if k not in obs:
        raise Exception(f'{k} not found in observation.')

  def build_update_ops(self, obs, targets):
    """
    This function will only be called once to create a TF graph which
    will be run repeatedly during training at the learner.

    All the arguments are tf placeholders (or nested structures of placeholders).

    Args:
      obs: [B, ...]
      targets: [B, N], N is the max node size.
    """
    self._validate_observations(obs)
    obs = ConfigDict(**obs)
    with tf.variable_scope(self._name):
      # flatten graph_features
      obs.graph_features = flatten_graphs(
          gn.graphs.GraphsTuple(**obs.graph_features))

      with tf.variable_scope('target_logits'):
        preds, logits_logged_vals = self._model.get_logits_and_next_state(obs)

      with tf.variable_scope('loss'):
        loss = tf.reduce_sum(obs.node_mask * ((preds - targets)**2))
        loss /= tf.reduce_sum(obs.node_mask)

      with tf.variable_scope('optimize'):
        opt_vals = self._optimize(loss)

      with tf.variable_scope('logged_vals'):
        self._logged_values = {
            'loss/supervised_loss': tf.reduce_sum(loss),
            **opt_vals,
            **logits_logged_vals,
            **self._extract_logged_values(obs),
        }

  def update(self, sess, feed_dict, profile_kwargs):
    """profile_kwargs pass to sess.run for profiling purposes."""
    _, vals = sess.run([self._train_op, self._logged_values],
                       feed_dict=feed_dict,
                       **profile_kwargs)
    return vals
