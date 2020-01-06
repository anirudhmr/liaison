import os
import pickle

import numpy as np
import tree as nest
from liaison.daper.dataset_constants import NORMALIZATION_CONSTANTS
from liaison.daper.milp.primitives import IntegerVariable
from liaison.env.utils.rins import *
from liaison.specs import ArraySpec
from liaison.utils import ConfigDict
from pyscipopt import Model


class Env:
  """
    RINS supervised environment.
    Use the optimal solution to fit given graph features.
  """
  # fields present only in variable nodes.
  # optimal lp solution (continuous relaxation)
  VARIABLE_LP_SOLN_FIELD = 0
  # input feasible solution.
  VARIABLE_FEASIBLE_SOLN_FIELD = 0
  # If the variable is constrainted to be an integer
  # as opposed to continuous variables.
  VARIABLE_IS_INTEGER_FIELD = 1
  # TODO: Add variable lower and upper bound vals.

  N_VARIABLE_FIELDS = 2

  # constant used in the constraints.
  CONSTRAINT_CONSTANT_FIELD = 0
  N_CONSTRAINT_FIELDS = 1

  # objective
  # current lp bound
  OBJ_LP_VALUE_FIELD = 0

  N_OBJECTIVE_FIELDS = 1

  # coefficient in the constraint as edge weight.
  EDGE_WEIGHT_FIELD = 0

  N_EDGE_FIELDS = 1

  GLOBAL_DUMMY_FIELD = 0

  N_GLOBAL_FIELDS = 1

  def __init__(self,
               id,
               seed,
               graph_seed=-1,
               graph_start_idx=0,
               n_graphs=1,
               dataset='milp-facilities-3',
               dataset_type='train',
               max_nodes=-1,
               max_edges=-1,
               **env_config):
    """If graph_seed < 0, then use the environment seed.
       max_nodes, max_edges -> Use for padding
    """
    self.config = ConfigDict(env_config)
    self.id = id
    self.seed = seed
    self._max_nodes = max_nodes
    self._max_edges = max_edges
    self.set_seed(seed)
    if graph_seed < 0: graph_seed = seed
    self._setup_graph_random_state(graph_seed)
    self._dataset = dataset
    self._dataset_type = dataset_type
    self._n_graphs = n_graphs
    self._graph_start_idx = graph_start_idx

    self.config.update(NORMALIZATION_CONSTANTS[dataset])

  def _sample(self, choice=None):
    n_graphs = self._n_graphs
    graph_start_idx = self._graph_start_idx

    if choice is None:
      choice = np.random.choice(
          range(graph_start_idx, graph_start_idx + n_graphs))

    self._milp_choice = choice
    return get_sample(self._dataset, self._dataset_type, choice)

  def _observation_graphnet_inductive(self, nodes):
    graph_features = dict(nodes=nodes,
                          edges=self._edges,
                          globals=self._globals,
                          senders=np.array(self._senders, dtype=np.int32),
                          receivers=np.array(self._receivers, dtype=np.int32),
                          n_node=np.array(len(nodes), dtype=np.int32),
                          n_edge=np.array(len(self._edges), dtype=np.int32))

    node_mask = np.zeros(len(nodes), dtype=np.int32)
    node_mask[0:len(self._variable_nodes
                    )] = self._variable_nodes[:, Env.VARIABLE_MASK_FIELD]

    node_mask = pad_last_dim(node_mask, self._max_nodes)
    graph_features = self._pad_graph_features(graph_features)
    obs = dict(graph_features=graph_features,
               node_mask=node_mask,
               mask=node_mask)
    return obs

  def next_sample(self):
    milp = self.milp = self._sample()
    var_names = list(milp.mip.varname2var.keys())
    # first construct variable nodes
    variable_nodes = np.zeros((len(var_names), Env.N_VARIABLE_FIELDS),
                              dtype=np.float32)
    # mask all integral variables
    feas_sol = [milp.feasible_solution[v] for v in var_names]
    variable_nodes[:, Env.VARIABLE_FEASIBLE_SOLN_FIELD] = feas_sol
    variable_nodes[:, Env.VARIABLE_LP_SOLN_FIELD] = [
        milp.optimal_lp_sol[v] for v in var_names
    ]
    variable_nodes[:, Env.VARIABLE_IS_INTEGER_FIELD] = [
        isinstance(milp.mip.varname2var[var_name], IntegerVariable)
        for var_name in var_names
    ]

    # now construct constraint nodes
    constraint_nodes = np.zeros(
        (len(milp.mip.constraints), Env.N_CONSTRAINT_FIELDS), dtype=np.float32)
    for i, c in enumerate(milp.mip.constraints):
      c = c.cast_sense_to_le()
      constraint_nodes[
          i, Env.
          CONSTRAINT_CONSTANT_FIELD] = c.rhs / self.config.constraint_rhs_normalizer

    objective_nodes = np.zeros((1, Env.N_OBJECTIVE_FIELDS), dtype=np.float32)
    objective_nodes[:,
                    0] = milp.feasible_objective / self.config.obj_normalizer
    objective_nodes[:,
                    1] = milp.feasible_objective / self.config.obj_normalizer

    # get undirected edge representation first
    n_edges = sum([len(c) for c in milp.mip.constraints]) + len(milp.mip.obj)
    edges = np.zeros((n_edges, Env.N_EDGE_FIELDS), dtype=np.float32)
    senders = np.zeros((n_edges), dtype=np.int32)
    receivers = np.zeros((n_edges), dtype=np.int32)
    i = 0
    for cid, c in enumerate(milp.mip.constraints):
      c = c.cast_sense_to_le()
      for var_name, coeff in zip(c.expr.var_names, c.expr.coeffs):
        edges[
            i, Env.
            EDGE_WEIGHT_FIELD] = coeff / self.config.constraint_coeff_normalizer
        # sender is the variable
        senders[i] = var_names.index(var_name)
        # receiver is the constraint.
        receivers[i] = len(variable_nodes) + cid
        i += 1

    for j, (var_name, coeff) in enumerate(
        zip(milp.mip.obj.expr.var_names, milp.mip.obj.expr.coeffs)):
      edges[i + j, Env.
            EDGE_WEIGHT_FIELD] = coeff / self.config.obj_coeff_normalizer
      senders[i + j] = var_names.index(var_name)
      receivers[i + j] = len(variable_nodes) + len(constraint_nodes)

    # now duplicate the edges to make them directed.
    edges = np.vstack((edges, edges))
    senders, receivers = np.hstack((senders, receivers)), np.hstack(
        (receivers, senders))

    globals_ = np.zeros(Env.N_GLOBAL_FIELDS, dtype=np.float32)

    self._edges = edges
    self._globals = globals_
    self._senders, self._receivers = senders, receivers
    self._variable_nodes = variable_nodes
    self._objective_nodes = objective_nodes
    self._constraint_nodes = constraint_nodes
    n_nodes = len(variable_nodes) + len(objective_nodes) + len(
        constraint_nodes)

    # compute masks for each node type
    self._var_type_mask = np.zeros((n_nodes), dtype=np.int32)
    self._var_type_mask[0:len(variable_nodes)] = 1
    self._constraint_type_mask = np.zeros((n_nodes), dtype=np.int32)
    self._constraint_type_mask[len(variable_nodes):len(variable_nodes) +
                               len(constraint_nodes)] = 1
    self._obj_type_mask = np.zeros((n_nodes), dtype=np.int32)
    self._obj_type_mask[len(variable_nodes) +
                        len(constraint_nodes):len(variable_nodes) +
                        len(constraint_nodes) + len(objective_nodes)] = 1
    variable_nodes = self._variable_nodes
    constraint_nodes = self._constraint_nodes
    objective_nodes = self._objective_nodes

    nodes = np.zeros(
        (len(variable_nodes) + len(constraint_nodes) + len(objective_nodes),
         max(Env.N_VARIABLE_FIELDS, Env.N_CONSTRAINT_FIELDS,
             Env.N_OBJECTIVE_FIELDS)),
        dtype=np.float32)

    nodes[0:len(variable_nodes), 0:Env.N_VARIABLE_FIELDS] = variable_nodes

    constraint_node_offset = len(variable_nodes)
    nodes[constraint_node_offset:constraint_node_offset +
          len(constraint_nodes), 0:Env.N_CONSTRAINT_FIELDS] = constraint_nodes

    objective_node_offset = constraint_node_offset + len(constraint_nodes)
    nodes[objective_node_offset:objective_node_offset +
          len(objective_nodes), 0:Env.N_OBJECTIVE_FIELDS] = objective_nodes

    if self.config.make_obs_for_mlp:
      obs = self._observation_mlp(nodes)
    elif self.config.make_obs_for_self_attention:
      obs = self._observation_self_attention(nodes)
    else:
      obs = self._observation_graphnet_inductive(nodes)

    # masks should be mutually disjoint.
    assert not np.any(
        np.logical_and(self._var_type_mask, self._constraint_type_mask))
    assert not np.any(np.logical_and(self._var_type_mask, self._obj_type_mask))
    assert not np.any(
        np.logical_and(self._constraint_type_mask, self._obj_type_mask))
    assert np.all(
        np.logical_or(
            self._var_type_mask,
            np.logical_or(self._constraint_type_mask, self._obj_type_mask)))

    obs = dict(
        **obs,
        var_type_mask=pad_last_dim(self._var_type_mask, self._max_nodes),
        constraint_type_mask=pad_last_dim(self._constraint_type_mask,
                                          self._max_nodes),
        obj_type_mask=pad_last_dim(self._obj_type_mask, self._max_nodes),
    )
    preds = pad_last_dim([milp.optimal_solution[v] for v in var_names],
                         self._max_nodes)
    return obs, preds

  def observation_spec(self):
    # Note: this function will be called before reset call is issued and
    # observation might not be ready.
    # But for this specific environment it works.
    obs, preds = self._next_sample()

    def mk_spec(path_tuple, np_arr):
      return ArraySpec(np_arr.shape,
                       np_arr.dtype,
                       name='_'.join(path_tuple) + '_spec')

    return nest.map_structure_with_tuple_paths(mk_spec, obs, preds)

  def _pad_graph_features(self, features: dict):
    features = ConfigDict(**features)
    features.update(nodes=pad_first_dim(features.nodes, self._max_nodes),
                    edges=pad_first_dim(features.edges, self._max_edges),
                    senders=pad_first_dim(features.senders, self._max_edges),
                    receivers=pad_first_dim(features.receivers,
                                            self._max_edges))
    return dict(**features)

  def set_seed(self, seed):
    np.random.seed(seed + self.id)

  def _setup_graph_random_state(self, seed):
    self._graph_random_state = np.random.RandomState(seed=seed)
