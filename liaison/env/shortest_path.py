import copy

import graph_nets as gn
import networkx as nx
import numpy as np
from liaison.env import Env as BaseEnv
from liaison.env.environment import restart, termination, transition
from liaison.env.utils.shortest_path import *
from liaison.specs import ArraySpec, BoundedArraySpec
from liaison.utils import ConfigDict
from tensorflow.contrib.framework import nest


class Env(BaseEnv):
  """
    Shortest path environment.
    At each step predict the next node in the path to the target.
    Similar to the rat-maze problem without obstacles.
  """
  # static
  NODE_IS_SRC_FIELD = 0
  NODE_IS_TARGET_FIELD = 1
  # dynamic
  NODE_VISITED_COUNT_FIELD = 2
  NODE_PREV_NODE_FIELD = 3
  NODE_CUR_NODE_FIELD = 4
  NODE_MASK_FIELD = 5

  N_NODE_FIELDS = 6

  # static
  EDGE_WEIGHT_FIELD = 0
  # dynamic
  EDGE_VISITED_FIELD = 1

  N_EDGE_FIELDS = 2

  # dynamic
  GLOBAL_STEP_COUNT_FIELD = 0
  GLOBAL_RETURNS_FIELD = 1

  N_GLOBAL_FIELDS = 2

  def __init__(self, id, seed, discount=1.0, graph_seed=-1, **env_config):
    """if graph_seed < 0, then use the environment seed"""
    self.config = ConfigDict(env_config)
    self.id = id
    self.seed = seed
    self.discount = discount
    self.set_seed(seed)

    if graph_seed < 0:
      graph_seed = seed
    # generate graph with 32 nodes.
    nx_graph, self._path = generate_networkx_graph(graph_seed, [32, 33])
    nx_graph = nx_graph.to_directed()
    # max number of steps in an episode.
    self._max_steps = 3 * len(nx_graph)
    self._nx_graph = nx_graph
    self._src_node = self._path[0]
    self._target_node = self._path[-1]
    self._shortest_path_length = sum([
        nx_graph[u][v][DISTANCE_WEIGHT_NAME] for u, v in pairwise(self._path)
    ])
    self._reset_graph_features = self._networkx_to_graph_features(
        nx_graph, self._src_node, self._target_node)
    self._graph_features = copy.deepcopy(self._reset_graph_features)

    self._curr_node = self._src_node
    self._reset_next_step = True

  def _networkx_to_graph_features(self, nx_graph, src_node, target_node):
    # Create graph features from the networkx graph
    # make sure to set all the static fields in the created features.
    # also initialize the dynamic fields to the right values.
    nodes = np.zeros((len(nx_graph), Env.N_NODE_FIELDS), dtype=np.float32)
    # static fields
    nodes[src_node, Env.NODE_IS_SRC_FIELD] = 1
    nodes[src_node, Env.NODE_CUR_NODE_FIELD] = 1
    nodes[target_node, Env.NODE_IS_TARGET_FIELD] = 1
    # dynamic fields
    nodes[:, Env.NODE_MASK_FIELD] = self._get_mask(src_node)

    edges = np.zeros([nx_graph.number_of_edges(), Env.N_EDGE_FIELDS],
                     dtype=np.float32)
    weights = nx.get_edge_attributes(nx_graph, DISTANCE_WEIGHT_NAME)
    for i, edge in enumerate(nx_graph.edges()):
      edges[i][Env.EDGE_WEIGHT_FIELD] = weights[edge]

    senders, receivers = zip(*nx_graph.edges())
    graph = gn.graphs.GraphsTuple(nodes=nodes,
                                  edges=edges,
                                  globals=np.zeros(Env.N_GLOBAL_FIELDS,
                                                   dtype=np.float32),
                                  senders=np.array(senders, dtype=np.int32),
                                  receivers=np.array(receivers,
                                                     dtype=np.int32),
                                  n_node=np.array(len(nodes), dtype=np.int32),
                                  n_edge=np.array(len(edges), dtype=np.int32))
    return graph

  def reset(self):
    self._reset_next_step = False
    self._ep_return = 0
    self._n_steps = 0
    # start from src node.
    self._curr_node = self._src_node
    self._graph_features = copy.deepcopy(self._reset_graph_features)

    return restart(self._observation())

  def set_seed(self, seed):
    np.random.seed(seed + self.id)

  def _get_mask(self, node):
    # return the mask with neighbors bits set to 1
    mask = np.zeros((len(self._nx_graph)), dtype=np.int32)
    mask[list(self._nx_graph.neighbors(node))] = 1
    assert np.sum(mask) > 0
    return mask

  def _observation_mlp(self):
    mask = np.int32(self._graph_features.nodes[:, Env.NODE_MASK_FIELD])
    obs = dict(features=self._graph_features.nodes.flatten(), mask=mask)
    return obs

  def _observation_graphnet_semi_supervised(self):
    # set ans as part of the node feature to the exact next node
    # too much of a give-away
    ans = np.zeros(len(self._graph_features.nodes), dtype=np.float32)
    ans[self._path] = 1

    graph_features = copy.deepcopy(self._graph_features)
    node_labels = np.arange(len(graph_features.nodes), dtype=np.float32)

    is_in_path = np.zeros((len(graph_features.edges), ), np.float32)
    for u, v in pairwise(self._path):
      for i, (s, r) in enumerate(
          zip(graph_features.senders, graph_features.receivers)):
        if (s == u and v == r):
          is_in_path[i] = 1

    ans = np.expand_dims(ans, -1)
    node_labels = np.expand_dims(node_labels, -1)
    is_in_path = np.expand_dims(is_in_path, -1)
    graph_features = graph_features.replace(
        nodes=np.hstack([graph_features.nodes, node_labels, ans]),
        edges=np.hstack([graph_features.edges, is_in_path]))

    mask = np.int32(self._graph_features.nodes[:, Env.NODE_MASK_FIELD])
    obs = dict(graph_features=dict(graph_features._asdict()), node_mask=mask)
    return obs

  def _observation_graphnet_inductive(self):
    mask = np.int32(self._graph_features.nodes[:, Env.NODE_MASK_FIELD])
    obs = dict(graph_features=dict(self._graph_features._asdict()),
               node_mask=mask)
    return obs

  def _observation(self):
    if self.config.make_obs_for_mlp:
      return self._observation_mlp()
    elif self.config.make_obs_for_graphnet_semi_supervised:
      return self._observation_graphnet_semi_supervised()
    else:
      return self._observation_graphnet_inductive()

  def step(self, action):
    if self._reset_next_step:
      return self.reset()

    # action is the next node in the path.
    action = int(action)
    # check if the previous step's mask was successfully applied.
    assert action in self._nx_graph.neighbors(self._curr_node)

    self._n_steps += 1
    graph_features = self._graph_features
    nx_graph = self._nx_graph

    ep_terminate = False
    # default reward of edge weight per step.
    rew = -1.0 * nx_graph[self._curr_node][action][
        DISTANCE_WEIGHT_NAME] / self._shortest_path_length
    # calculate reward for action
    if action == self._target_node:
      ep_terminate = True
      # net reward for an optimal episode = 0
      rew = 1
    elif self._n_steps == self._max_steps:
      ep_terminate = True

    self._ep_return += np.power(self.discount, self._n_steps - 1) * rew

    # update graph features based on the action.
    nodes = graph_features.nodes
    edges = graph_features.edges
    globals_ = graph_features.globals

    # update dynamic fields of the nodes
    nodes[action, Env.NODE_VISITED_COUNT_FIELD] += 1
    nodes[:, Env.NODE_PREV_NODE_FIELD] = 0
    nodes[self._curr_node, Env.NODE_PREV_NODE_FIELD] = 1
    nodes[:, Env.NODE_CUR_NODE_FIELD] = 0
    nodes[action, Env.NODE_CUR_NODE_FIELD] = 1
    nodes[:, Env.NODE_MASK_FIELD] = self._get_mask(action)

    # update edge dynamic fields
    edges[list(self._nx_graph.edges()).index(
        (self._curr_node, action)), Env.EDGE_VISITED_FIELD] += 1

    # update global dynamic fields
    globals_[Env.GLOBAL_STEP_COUNT_FIELD] = self._n_steps
    globals_[Env.GLOBAL_RETURNS_FIELD] = self._ep_return

    self._graph_features = graph_features.replace(nodes=nodes,
                                                  edges=edges,
                                                  globals=globals_)

    self._curr_node = action
    if ep_terminate:
      self._reset_next_step = True
      return termination(rew, self._observation())
    else:
      return transition(rew, self._observation())

  def observation_spec(self):
    # Note: this function will be called before reset call is issued and
    # observation might not be ready.
    # But for this specific environment it works.
    obs = self._observation()

    def mk_spec(path_tuple, np_arr):
      return ArraySpec(np_arr.shape,
                       np_arr.dtype,
                       name='_'.join(path_tuple) + '_spec')

    return nest.map_structure_with_tuple_paths(mk_spec, obs)

  def action_spec(self):
    return BoundedArraySpec((),
                            np.int32,
                            minimum=0,
                            maximum=len(self._nx_graph) - 1,
                            name='action_spec')
