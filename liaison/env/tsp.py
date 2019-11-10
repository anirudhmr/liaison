import copy
import os
import pickle

import graph_nets as gn
import networkx as nx
import numpy as np
import scipy
from liaison.daper.dataset_constants import DATASET_PATH, LENGTH_MAP
from liaison.env import Env as BaseEnv
from liaison.env.environment import restart, termination, transition
from liaison.env.utils.shortest_path import *
from liaison.specs import ArraySpec, BoundedArraySpec
from liaison.utils import ConfigDict
from tensorflow.contrib.framework import nest


class Env(BaseEnv):
  """
    Travelling salesman environment.
    At each step predict the next node in the path back to the source.
    Similar to the rat-maze problem without obstacles.
  """
  # static
  NODE_IS_SRC_FIELD = 0

  # dynamic
  # whether node is visited or not in this episode.
  NODE_VISITED_FIELD = 1
  # whether this is the current node
  NODE_CUR_NODE_FIELD = 2
  # mask bits
  NODE_MASK_FIELD = 3
  # x coord of the node
  NODE_X_FIELD = 4
  # y coord of the node
  NODE_Y_FIELD = 5
  # Normalized path index of the node.
  NODE_PATH_INDEX_FIELD = 6

  N_NODE_FIELDS = 7

  # static
  # edge distance as the cost.
  EDGE_WEIGHT_FIELD = 0
  # dynamic
  # whether the edge has been visited or not.
  EDGE_VISITED_FIELD = 1
  # The edge's path index (normalized)
  EDGE_PATH_INDEX_FIELD = 2

  N_EDGE_FIELDS = 3

  # dynamic
  GLOBAL_STEP_COUNT_FIELD = 0
  # counts the total return of the episode so far.
  GLOBAL_RETURNS_FIELD = 1

  N_GLOBAL_FIELDS = 2

  def __init__(self,
               id,
               seed,
               graph_seed=-1,
               graph_idx=0,
               dataset='tsp-20',
               dataset_type='train',
               **env_config):
    """if graph_seed < 0, then use the environment seed"""
    self.config = ConfigDict(env_config)
    self.id = id
    self.seed = seed
    self.set_seed(seed)
    if graph_seed < 0: graph_seed = seed
    self._setup_graph_random_state(graph_seed)

    # generate graph with 32 nodes.
    self._src_node, self._fixed_graph_features, self._distance_matrix, self._tsp = self._load_graph(
        dataset, dataset_type, graph_idx)

    # call reset so that obs_spec can work without calling reset
    self._ep_return = -10
    self.reset()

  def _load_graph(self, dataset, dataset_type, graph_idx):
    graph_idx = int(graph_idx)
    path = DATASET_PATH[dataset]
    assert graph_idx < LENGTH_MAP[dataset][dataset_type]
    with open(os.path.join(path, dataset_type, '%d.pkl' % graph_idx),
              'rb') as f:
      tsp = pickle.load(f)

    # sample src node.
    src_node = self._graph_random_state.randint(0, len(tsp))

    n_nodes = len(tsp.locs)
    nodes = np.zeros((n_nodes, Env.N_NODE_FIELDS), dtype=np.float32)
    nodes[src_node, Env.NODE_IS_SRC_FIELD] = 1
    nodes[src_node, Env.NODE_CUR_NODE_FIELD] = 1
    nodes[src_node, Env.NODE_VISITED_FIELD] = 1
    nodes[src_node, Env.NODE_PATH_INDEX_FIELD] = 1.0 / n_nodes

    # set mask for all nodes except src node.
    nodes[:, Env.NODE_MASK_FIELD] = 1
    nodes[src_node, Env.NODE_MASK_FIELD] = 0

    nodes[:, Env.NODE_X_FIELD] = tsp.locs[:, 0]
    nodes[:, Env.NODE_Y_FIELD] = tsp.locs[:, 1]

    # edges[i * n_node + j] = edge from ith node to jth node.
    edges = np.zeros((n_nodes * n_nodes, Env.N_EDGE_FIELDS), dtype=np.float32)
    dst = scipy.spatial.distance_matrix(tsp.locs, tsp.locs)
    edges[:, Env.EDGE_WEIGHT_FIELD] = dst.flatten()

    globals_ = np.zeros(Env.N_GLOBAL_FIELDS)
    senders, receivers = list(
        zip(*[(i, j) for i in range(n_nodes) for j in range(n_nodes)]))

    return src_node, gn.graphs.GraphsTuple(
        nodes=nodes,
        edges=edges,
        globals=globals_,
        senders=np.array(senders, dtype=np.int32),
        receivers=np.array(receivers, dtype=np.int32),
        n_node=np.array(len(nodes), dtype=np.int32),
        n_edge=np.array(len(edges), dtype=np.int32)), dst, tsp

  def reset(self):
    self._reset_next_step = False
    self._prev_ep_return = self._ep_return
    self._ep_return = 0
    self._n_steps = 0
    # start from src node.
    self._curr_node = self._src_node
    # reset features
    self._graph_features = copy.deepcopy(self._fixed_graph_features)
    # path collected so far
    self._path = [self._curr_node]
    return restart(self._observation())

  def _observation_mlp(self):
    mask = np.int32(self._graph_features.nodes[:, Env.NODE_MASK_FIELD])
    obs = dict(features=self._graph_features.nodes.flatten(), mask=mask)
    return obs

  def _observation_graphnet_inductive(self):
    mask = np.int32(self._graph_features.nodes[:, Env.NODE_MASK_FIELD])
    obs = dict(graph_features=dict(self._graph_features._asdict()),
               node_mask=mask)
    return obs

  def _observation(self):
    if self.config.make_obs_for_mlp:
      obs = self._observation_mlp()
    else:
      obs = self._observation_graphnet_inductive()

    obs['log_values'] = dict(ep_return=np.float32(self._prev_ep_return))
    return obs

  def step(self, action):
    if self._reset_next_step:
      return self.reset()

    self._n_steps += 1
    graph_features = self._graph_features
    # action is the next node in the path.
    action = int(action)
    mask = graph_features.nodes[:, Env.NODE_MASK_FIELD]
    # check if the previous step's mask was successfully applied
    assert mask[action], (mask, action)

    # update graph features based on the action.
    nodes = graph_features.nodes
    edges = graph_features.edges
    globals_ = graph_features.globals

    # default reward of edge weight per step.
    rew = -1.0 * self._distance_matrix[self._curr_node][
        action] / self._tsp.baseline_results.concorde.objective

    self._ep_return += rew
    # update dynamic fields of the nodes
    nodes[action, Env.NODE_VISITED_FIELD] += 1
    nodes[:, Env.NODE_CUR_NODE_FIELD] = 0
    nodes[action, Env.NODE_CUR_NODE_FIELD] = 1
    nodes[action, Env.NODE_MASK_FIELD] = 0

    if np.sum(nodes[:, Env.NODE_MASK_FIELD]) == 0:
      assert len(self._path) == len(nodes) - 1
      # if mask is 0 then the next step will be reset. so set an arbitrary mask
      nodes[:, Env.NODE_MASK_FIELD] = 1

    # edges[i * n_node + j] = edge from ith node to jth node.
    edges[self._curr_node * len(nodes) + action, Env.EDGE_VISITED_FIELD] += 1

    # update global dynamic fields
    globals_[Env.GLOBAL_STEP_COUNT_FIELD] = self._n_steps
    globals_[Env.GLOBAL_RETURNS_FIELD] = self._ep_return

    self._graph_features = graph_features.replace(nodes=nodes,
                                                  edges=edges,
                                                  globals=globals_)

    self._curr_node = action
    self._path.append(self._curr_node)

    if len(self._path) == len(nodes):
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
                            maximum=len(self._graph_features.nodes) - 1,
                            name='action_spec')

  def set_seed(self, seed):
    np.random.seed(seed + self.id)

  def _setup_graph_random_state(self, seed):
    self._graph_random_state = np.random.RandomState(seed=seed)
