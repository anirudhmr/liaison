import copy
import os
import pickle
from math import fabs
from typing import Any, Dict, Text, Tuple, Union

import graph_nets as gn
import liaison.utils as U
import networkx as nx
import numpy as np
import scipy
import tree as nest
from liaison.daper.dataset_constants import (DATASET_PATH, LENGTH_MAP,
                                             NORMALIZATION_CONSTANTS)
from liaison.daper.milp.primitives import (ContinuousVariable, IntegerVariable,
                                           MIPInstance)
from liaison.env import Env as BaseEnv
from liaison.env.environment import restart, termination, transition
from liaison.env.utils.rins import *
from liaison.specs import ArraySpec, BoundedArraySpec
from liaison.utils import ConfigDict
from pyscipopt import SCIP_PARAMSETTING, Model


class Env(BaseEnv):
  """
    RINS environment.
    At each step predict the variable to unfix.
    After k steps solve the sub-MIP problem formed by the unfixed variables.
    Use the objective value of the solved sub-MIP as reward signal.
  """
  # fields present only in variable nodes.
  # mask bits
  VARIABLE_MASK_FIELD = 0
  # current integral assignment
  VARIABLE_CURR_ASSIGNMENT_FIELD = 1
  # current lp solution (if unfixed).
  # if node fixed then this field is equal to the current assignment.
  VARIABLE_LP_SOLN_FIELD = 2
  # If the variable is constrainted to be an integer
  # as opposed to continuous variables.
  VARIABLE_IS_INTEGER_FIELD = 3
  # optimal lp solution (for the continuous relaxation)
  VARIABLE_OPTIMAL_LP_SOLN_FIELD = 4

  # see Learning to Branch in Mixed Integer Programming paper from Khalil et al.,
  # for a description of some of the following features.
  VARIABLE_OBJ_COEFF_FIELD = 5

  VARIABLE_N_CONSTRAINTS_FIELD = 6

  # stats of the constraints that the variable is involved in.
  VARIABLE_STATS_CONSTRAINT_DEGREES_MEAN_FIELD = 7
  VARIABLE_STATS_CONSTRAINT_DEGREES_STD_FIELD = 8
  VARIABLE_STATS_CONSTRAINT_DEGREES_MIN_FIELD = 9
  VARIABLE_STATS_CONSTRAINT_DEGREES_MAX_FIELD = 10

  # slack is distance from the floor
  # ceil is distance from the roof
  VARIABLE_LP_SOLN_SLACK_DOWN_FIELD = 11
  VARIABLE_LP_SOLN_SLACK_UP_FIELD = 12
  VARIABLE_OPTIMAL_LP_SOLN_SLACK_DOWN_FIELD = 13
  VARIABLE_OPTIMAL_LP_SOLN_SLACK_UP_FIELD = 14
  # TODO: Add variable lower and upper bound vals.
  N_VARIABLE_FIELDS = 15

  # constant used in the constraints.
  CONSTRAINT_CONSTANT_FIELD = 0
  CONSTRAINT_DEGREE_FIELD = 1
  CONSTRAINT_STATS_COEFF_MEAN_FIELD = 3
  CONSTRAINT_STATS_COEFF_STD_FIELD = 4
  CONSTRAINT_STATS_COEFF_MIN_FIELD = 5
  CONSTRAINT_STATS_COEFF_MAX_FIELD = 6
  # TODO: Add dual solution for the constraint here.
  N_CONSTRAINT_FIELDS = 7

  # objective
  # current lp bound
  OBJ_LP_VALUE_FIELD = 0
  # current value achieved by
  # the integral solution.
  OBJ_INT_VALUE_FIELD = 1

  N_OBJECTIVE_FIELDS = 2

  # coefficient in the constraint as edge weight.
  # This edge weight is normalized by the global dataset normalization
  # constant
  EDGE_WEIGHT_FIELD = 0

  N_EDGE_FIELDS = 1

  GLOBAL_STEP_NUMBER = 0
  # number of unfixes left before sub-mip is solved
  GLOBAL_UNFIX_LEFT = 1
  GLOBAL_N_LOCAL_MOVES = 2
  GLOBAL_LOCAL_SEARCH_STEP = 3

  N_GLOBAL_FIELDS = 4

  def __init__(self,
               id,
               seed,
               graph_seed=-1,
               graph_start_idx=0,
               n_graphs=1,
               dataset='',
               dataset_type='train',
               k=5,
               n_local_moves=10,
               max_nodes=-1,
               max_edges=-1,
               enable_curriculum=False,
               **env_config):
    """If graph_seed < 0, then use the environment seed.
       k -> Max number of variables to unfix at a time.
            Informally, this is a bound on the local search
            neighbourhood size.
       max_nodes, max_edges -> Use for padding
    """
    self.config = ConfigDict(env_config)
    self.id = id
    self.k = k
    self._steps_per_episode = k * n_local_moves
    self.seed = seed
    self._max_nodes = max_nodes
    self._max_edges = max_edges
    self.set_seed(seed)
    if graph_seed < 0: graph_seed = seed
    self._setup_graph_random_state(graph_seed)
    self._dataset = dataset
    self._dataset_type = dataset_type
    if n_graphs >= LENGTH_MAP[dataset][dataset_type]:
      print(
          'WARNING: Specified number of graphs exceeds whats available in dataset_constants.py'
      )
      self._n_graphs = LENGTH_MAP[dataset][dataset_type]
    else:
      self._n_graphs = n_graphs
    self._graph_start_idx = graph_start_idx

    self.config.update(NORMALIZATION_CONSTANTS[dataset])
    # call reset so that obs_spec can work without calling reset
    self._ep_return = None
    self._prev_ep_return = np.nan
    self._prev_avg_quality = np.nan
    self._prev_best_quality = np.nan
    self._prev_final_quality = np.nan
    self._prev_mean_work = np.nan
    self._reset_next_step = True
    self._enable_curriculum = enable_curriculum
    if enable_curriculum:
      self._global_step_fetcher = GlobalStepFetcher()
    else:
      self._global_step_fetcher = None
    # map from sample to length of the mip
    self._sample_lengths = None
    self.reset()

  def _pick_sample_for_curriculum(self):
    n_graphs = self._n_graphs
    graph_start_idx = self._graph_start_idx
    config = self.config

    if self._sample_lengths is None:
      self._sample_lengths = {}
      for i in range(graph_start_idx, graph_start_idx + n_graphs):
        sample = get_sample(self._dataset, self._dataset_type, i)
        self._sample_lengths[i] = sample['problem_size']
    step = self._global_step_fetcher.get()
    val = config.sample_size_schedule.start_value
    if step >= config.sample_size_schedule.start_step:
      val += ((step - config.sample_size_schedule.start_step) *
              (config.sample_size_schedule.max_value -
               config.sample_size_schedule.start_value) /
              config.sample_size_schedule.dec_steps)
    val = min(val, config.sample_size_schedule.max_value)

    choices = []
    for i, length in self._sample_lengths.items():
      if length <= val:
        choices.append(i)
    return np.random.choice(choices)

  def _sample(self, choice=None):
    n_graphs = self._n_graphs
    graph_start_idx = self._graph_start_idx
    if choice is None:
      if self._enable_curriculum:
        choice = self._pick_sample_for_curriculum()
      else:
        choice = np.random.choice(
            range(graph_start_idx, graph_start_idx + n_graphs))

    self._milp_choice = choice
    return get_sample(self._dataset, self._dataset_type, choice)

  def _observation_mlp(self, nodes):
    mask = np.int32(self._variable_nodes[:, Env.VARIABLE_MASK_FIELD])
    features = np.hstack((self._variable_nodes.flatten(), self._globals))

    if self.config.mlp_embed_constraints:
      milp = self.milp
      var_names = self._var_names
      # constraint_features[i, j] = coefficient for the
      #                             jth variable in the ith constraint
      constraint_features = np.zeros(
          (len(milp.mip.constraints), len(var_names)), dtype=np.float32)

      for cid, c in enumerate(milp.mip.constraints):
        c = c.cast_sense_to_le()
        for var_name, coeff in zip(c.expr.var_names, c.expr.coeffs):
          constraint_features[cid, self._varnames2varidx[
              var_name]] = coeff / self.config.obj_coeff_normalizer

      features = np.hstack((features, constraint_features.flatten()))

    obs = dict(features=np.float32(features),
               mask=mask,
               mlp_mask=mask,
               globals=np.float32(self._globals))
    return obs

  def _observation_self_attention(self, nodes):
    mask = np.int32(self._variable_nodes[:, Env.VARIABLE_MASK_FIELD])
    features = np.hstack((nodes.flatten(), self._globals))
    cv = []
    for cid, c in enumerate(self.milp.mip.constraints):
      c = c.cast_sense_to_le()
      y = np.zeros(len(self._var_names), dtype=np.int32)
      for var_name, coeff in zip(c.expr.var_names, c.expr.coeffs):
        y[self._varnames2varidx[var_name]] = coeff
      cv.append(y)

    # var_embeddings[i][j] = c => ith variable uses jth constraint with coefficient c
    var_embeddings = np.transpose(np.asarray(cv, dtype=np.float32))

    obs = dict(mask=mask,
               var_nodes=np.float32(self._variable_nodes),
               var_embeddings=var_embeddings,
               globals=np.float32(self._globals))
    return obs

  def _observation_graphnet_inductive(self, nodes):
    graph_features = dict(nodes=nodes,
                          globals=np.array(self._globals, dtype=np.float32),
                          n_node=np.array(len(nodes), dtype=np.int32),
                          **self._static_graph_features[0])

    if self.config.attach_node_labels:
      labels = np.eye(len(nodes), dtype=np.float32)
      nodes = np.hstack((nodes, labels))
      graph_features.update(nodes=nodes)

    node_mask = np.zeros(len(nodes), dtype=np.int32)
    node_mask[0:len(self._variable_nodes
                    )] = self._variable_nodes[:, Env.VARIABLE_MASK_FIELD]

    node_mask = pad_last_dim(node_mask, self._max_nodes)
    graph_features = self._pad_graph_features(graph_features)
    obs = dict(graph_features=graph_features,
               node_mask=node_mask,
               mask=node_mask,
               **self._static_graph_features[1])
    return obs

  def _observation_bipartite_graphnet(self):
    graph_features = dict(left_nodes=pad_first_dim(self._variable_nodes,
                                                   self._max_nodes),
                          right_nodes=pad_first_dim(self._constraint_nodes,
                                                    self._max_nodes),
                          n_left_nodes=np.int32(len(self._variable_nodes)),
                          n_right_nodes=np.int32(len(self._constraint_nodes)),
                          globals=np.array(self._globals, dtype=np.float32),
                          **self._static_graph_features)

    if self.config.attach_node_labels:
      left_labels = np.eye(len(graph_features.left_nodes), dtype=np.float32)
      left_nodes = np.hstack((graph_features.left_nodes, left_labels))

      right_labels = np.eye(len(graph_features.right_nodes), dtype=np.float32)
      right_nodes = np.hstack((graph_features.right_nodes, right_labels))

      graph_features.update(left_nodes=left_nodes, right_nodes=right_nodes)

    mask = pad_last_dim(self._variable_nodes[:, Env.VARIABLE_MASK_FIELD],
                        self._max_nodes)
    return dict(graph_features=graph_features, node_mask=mask, mask=mask)

  def _observation(self):
    variable_nodes = self._variable_nodes
    constraint_nodes = self._constraint_nodes
    objective_nodes = self._objective_nodes

    nodes = np.zeros(
        (len(variable_nodes) + len(constraint_nodes) + len(objective_nodes),
         variable_nodes.shape[1] + constraint_nodes.shape[1] +
         objective_nodes.shape[1]),
        dtype=np.float32)

    nodes[0:len(variable_nodes), :variable_nodes.shape[1]] = variable_nodes

    constraint_node_offset = len(variable_nodes)
    nodes[constraint_node_offset:constraint_node_offset +
          len(constraint_nodes), :constraint_nodes.shape[1]] = constraint_nodes

    objective_node_offset = constraint_node_offset + len(constraint_nodes)
    nodes[objective_node_offset:objective_node_offset +
          len(objective_nodes), :objective_nodes.shape[1]] = objective_nodes

    obs = {}
    if self.config.make_obs_for_mlp:
      obs.update(self._observation_mlp(nodes))

    if self.config.make_obs_for_self_attention:
      obs.update(self._observation_self_attention(nodes))

    if self.config.make_obs_for_graphnet:
      obs.update(self._observation_graphnet_inductive(nodes))

    if self.config.make_obs_for_bipartite_graphnet:
      obs.update(self._observation_bipartite_graphnet())

    obs = dict(
        **obs,
        optimal_solution=pad_last_dim(self._optimal_soln, self._max_nodes),
        log_values=dict(  # useful for tensorboard.
            ep_return=np.float32(self._prev_ep_return),
            avg_quality=np.float32(self._prev_avg_quality),
            best_quality=np.float32(self._prev_best_quality),
            final_quality=np.float32(self._prev_final_quality),
            mip_work=np.float32(self._prev_mean_work),
        ),
        curr_episode_log_values=dict(
            ep_return=np.float32(self._ep_return),
            avg_quality=np.float32(np.mean(self._qualities)),
            best_quality=np.float32(self._best_quality),
            final_quality=np.float32(self._final_quality),
            **nest.map_structure(np.float32, dict(self._mip_stats))))
    return obs

  def _encode_static_graph_features(self):
    milp = self.milp
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
        senders[i] = self._varnames2varidx[var_name]
        # receiver is the constraint.
        receivers[i] = len(self._variable_nodes) + cid
        i += 1

    for j, (var_name, coeff) in enumerate(
        zip(milp.mip.obj.expr.var_names, milp.mip.obj.expr.coeffs)):
      edges[i + j, Env.
            EDGE_WEIGHT_FIELD] = coeff / self.config.obj_coeff_normalizer
      senders[i + j] = self._varnames2varidx[var_name]
      receivers[i +
                j] = len(self._variable_nodes) + len(self._constraint_nodes)

    # now duplicate the edges to make them directed.
    edges = np.vstack((edges, edges))
    senders, receivers = np.hstack((senders, receivers)), np.hstack(
        (receivers, senders))

    # compute masks for each node type
    variable_nodes = self._variable_nodes
    objective_nodes = self._objective_nodes
    constraint_nodes = self._constraint_nodes
    n_nodes = len(variable_nodes) + len(objective_nodes) + len(
        constraint_nodes)
    var_type_mask = np.zeros((n_nodes), dtype=np.int32)
    var_type_mask[0:len(variable_nodes)] = 1
    constraint_type_mask = np.zeros((n_nodes), dtype=np.int32)
    constraint_type_mask[len(variable_nodes):len(variable_nodes) +
                         len(constraint_nodes)] = 1
    obj_type_mask = np.zeros((n_nodes), dtype=np.int32)
    obj_type_mask[len(variable_nodes) +
                  len(constraint_nodes):len(variable_nodes) +
                  len(constraint_nodes) + len(objective_nodes)] = 1

    # masks should be mutually disjoint.
    assert not np.any(np.logical_and(var_type_mask, constraint_type_mask))
    assert not np.any(np.logical_and(var_type_mask, obj_type_mask))
    assert not np.any(np.logical_and(constraint_type_mask, obj_type_mask))
    assert np.all(
        np.logical_or(var_type_mask,
                      np.logical_or(constraint_type_mask, obj_type_mask)))

    return dict(edges=edges,
                senders=senders,
                receivers=receivers,
                n_edge=np.int32(len(edges))), dict(
                    var_type_mask=pad_last_dim(var_type_mask, self._max_nodes),
                    constraint_type_mask=pad_last_dim(constraint_type_mask,
                                                      self._max_nodes),
                    obj_type_mask=pad_last_dim(obj_type_mask, self._max_nodes))

  def _encode_static_bipartite_graph_features(self):
    milp = self.milp
    # get undirected edge representation first
    n_edges = sum([len(c) for c in milp.mip.constraints])
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
        senders[i] = self._varnames2varidx[var_name]
        # receiver is the constraint.
        receivers[i] = cid
        i += 1
    return dict(edges=pad_first_dim(edges, self._max_edges),
                senders=pad_first_dim(senders, self._max_edges),
                receivers=pad_first_dim(receivers, self._max_edges),
                n_edge=np.int32(len(edges)))

  def reset(self):
    milp = self.milp = self._sample()
    self._ep_return = 0
    self._n_steps = 0
    self._n_local_moves = 0
    self._reset_next_step = False
    self._best_quality = self._primal_gap(milp.feasible_objective)
    self._final_quality = self._best_quality
    self._qualities = [self._best_quality]
    self._mip_works = []
    # mip stats for the current step
    self._mip_stats = ConfigDict(mip_work=0,
                                 n_cuts=0,
                                 n_cuts_applied=0,
                                 n_lps=0,
                                 solving_time=0.,
                                 pre_solving_time=0.,
                                 time_elapsed=0.)

    self._var_names = var_names = list(milp.mip.varname2var.keys())
    self._varnames2varidx = {}
    for i, var_name in enumerate(var_names):
      self._varnames2varidx[var_name] = i

    # optimal solution can be used for supervised auxiliary tasks.
    self._optimal_soln = np.float32(
        [milp.optimal_solution[v] for v in var_names])
    # first construct variable nodes
    variable_nodes = np.zeros((len(var_names), Env.N_VARIABLE_FIELDS),
                              dtype=np.float32)
    # mask all integral variables
    feas_sol = [milp.feasible_solution[v] for v in var_names]
    variable_nodes[:, Env.VARIABLE_CURR_ASSIGNMENT_FIELD] = feas_sol
    variable_nodes[:, Env.VARIABLE_LP_SOLN_FIELD] = feas_sol
    variable_nodes[:, Env.VARIABLE_LP_SOLN_SLACK_UP_FIELD] = np_slack_up(
        feas_sol)
    variable_nodes[:, Env.VARIABLE_LP_SOLN_SLACK_DOWN_FIELD] = np_slack_down(
        feas_sol)
    variable_nodes[:, Env.VARIABLE_IS_INTEGER_FIELD] = [
        isinstance(milp.mip.varname2var[var_name], IntegerVariable)
        for var_name in var_names
    ]
    variable_nodes[:, Env.VARIABLE_OPTIMAL_LP_SOLN_FIELD] = [
        milp.optimal_lp_sol[v] for v in var_names
    ]
    variable_nodes[:, Env.
                   VARIABLE_OPTIMAL_LP_SOLN_SLACK_DOWN_FIELD] = np_slack_down(
                       variable_nodes[:, Env.VARIABLE_OPTIMAL_LP_SOLN_FIELD])
    variable_nodes[:, Env.
                   VARIABLE_OPTIMAL_LP_SOLN_SLACK_UP_FIELD] = np_slack_up(
                       variable_nodes[:, Env.VARIABLE_OPTIMAL_LP_SOLN_FIELD])

    variable_nodes = self._reset_mask(variable_nodes)
    for var_name, coeff in zip(milp.mip.obj.expr.var_names,
                               milp.mip.obj.expr.coeffs):
      variable_nodes[
          self._varnames2varidx[var_name], Env.
          VARIABLE_OBJ_COEFF_FIELD] = coeff / self.config.obj_coeff_normalizer

    variable_stats_degrees = {}
    for i, cons in enumerate(milp.mip.constraints):
      for var in cons.expr.var_names:
        if var in variable_stats_degrees:
          variable_stats_degrees[var].append(len(cons))
        else:
          variable_stats_degrees[var] = [len(cons)]

    for i, var_name in self._varnames2varidx.items():
      l = variable_stats_degrees.get(var_name, None)
      # TODO: Normalize these fields.
      if l:
        variable_nodes[i, Env.
                       VARIABLE_STATS_CONSTRAINT_DEGREES_MAX_FIELD] = max(l)
        variable_nodes[
            i, Env.VARIABLE_STATS_CONSTRAINT_DEGREES_MEAN_FIELD] = np.mean(l)
        variable_nodes[i, Env.
                       VARIABLE_STATS_CONSTRAINT_DEGREES_STD_FIELD] = np.std(l)
        variable_nodes[i, Env.
                       VARIABLE_STATS_CONSTRAINT_DEGREES_MIN_FIELD] = min(l)

    # Normalization
    for field in [
        Env.VARIABLE_STATS_CONSTRAINT_DEGREES_MAX_FIELD,
        Env.VARIABLE_STATS_CONSTRAINT_DEGREES_MIN_FIELD,
        Env.VARIABLE_STATS_CONSTRAINT_DEGREES_MEAN_FIELD,
        Env.VARIABLE_STATS_CONSTRAINT_DEGREES_STD_FIELD,
    ]:
      if np.mean(variable_nodes[:, field]):
        variable_nodes[:, field] /= np.mean(variable_nodes[:, field])

    # now construct constraint nodes
    constraint_nodes = np.zeros(
        (len(milp.mip.constraints), Env.N_CONSTRAINT_FIELDS), dtype=np.float32)
    for i, c in enumerate(milp.mip.constraints):
      c = c.cast_sense_to_le()
      constraint_nodes[
          i, Env.
          CONSTRAINT_CONSTANT_FIELD] = c.rhs / self.config.constraint_rhs_normalizer
      # Normalize this?
      constraint_nodes[i, Env.CONSTRAINT_DEGREE_FIELD] = len(c)
      coeffs = list(c.expr.coeffs)
      constraint_nodes[i, Env.CONSTRAINT_STATS_COEFF_MEAN_FIELD] = np.mean(
          coeffs) / self.config.constraint_coeff_normalizer
      constraint_nodes[i, Env.CONSTRAINT_STATS_COEFF_MIN_FIELD] = np.min(
          coeffs) / self.config.constraint_coeff_normalizer
      constraint_nodes[i, Env.CONSTRAINT_STATS_COEFF_MAX_FIELD] = np.max(
          coeffs) / self.config.constraint_coeff_normalizer

    objective_nodes = np.zeros((1, Env.N_OBJECTIVE_FIELDS), dtype=np.float32)
    objective_nodes[:,
                    0] = milp.feasible_objective / self.config.obj_normalizer
    objective_nodes[:,
                    1] = milp.feasible_objective / self.config.obj_normalizer

    globals_ = np.zeros(Env.N_GLOBAL_FIELDS, dtype=np.float32)
    globals_[Env.GLOBAL_STEP_NUMBER] = self._n_steps / np.sqrt(
        self._steps_per_episode)
    globals_[Env.GLOBAL_UNFIX_LEFT] = self.k
    globals_[Env.GLOBAL_N_LOCAL_MOVES] = self._n_local_moves

    self._globals = globals_
    self._variable_nodes = variable_nodes
    self._objective_nodes = objective_nodes
    self._constraint_nodes = constraint_nodes

    self._unfixed_variables = set([
        var for var in var_names
        if isinstance(milp.mip.varname2var[var], ContinuousVariable)
    ])
    self._curr_soln = copy.deepcopy(milp.feasible_solution)
    self._curr_obj = milp.feasible_objective
    self._prev_obj = milp.feasible_objective

    # static features computed only once per episode.
    if self.config.make_obs_for_graphnet:
      self._static_graph_features = self._encode_static_graph_features()
    elif self.config.make_obs_for_bipartite_graphnet:
      self._static_graph_features = self._encode_static_bipartite_graph_features(
      )
    return restart(self._observation())

  def _scip_solve(self, mip):
    """solves a mip/lp using scip"""
    solver = Model()
    solver.hideOutput()
    if self.config.disable_maxcuts:
      for param in [
          'separating/maxcuts', 'separating/maxcutsroot',
          'propagating/maxrounds', 'propagating/maxroundsroot',
          'presolving/maxroundsroot'
      ]:
        solver.setIntParam(param, 0)

      solver.setBoolParam('conflict/enable', False)
      solver.setPresolve(SCIP_PARAMSETTING.OFF)

    solver.setBoolParam('randomization/permutevars', True)
    # seed is set to 0 permanently.
    solver.setIntParam('randomization/permutationseed', 0)
    solver.setIntParam('randomization/randomseedshift', 0)

    mip.add_to_scip_solver(solver)
    with U.Timer() as timer:
      solver.optimize()
    assert solver.getStatus() == 'optimal', solver.getStatus()
    obj = float(solver.getObjVal())
    ass = {var.name: solver.getVal(var) for var in solver.getVars()}
    mip_stats = ConfigDict(mip_work=solver.getNNodes(),
                           n_cuts=solver.getNCuts(),
                           n_cuts_applied=solver.getNCutsApplied(),
                           n_lps=solver.getNLPs(),
                           solving_time=solver.getSolvingTime(),
                           pre_solving_time=solver.getPresolvingTime(),
                           time_elapsed=timer.to_seconds())
    solver.freeProb()
    return ass, obj, mip_stats

  def _reset_mask(self, variable_nodes):
    variable_nodes[:, Env.
                   VARIABLE_MASK_FIELD] = variable_nodes[:, Env.
                                                         VARIABLE_IS_INTEGER_FIELD]
    return variable_nodes

  def _primal_gap(self, curr_obj):
    optimal_obj = self.milp.optimal_objective
    if curr_obj == 0 and optimal_obj == 0:
      rew = 1.
    elif np.sign(curr_obj) * np.sign(optimal_obj) < 0:
      rew = 1.
    else:
      rew = fabs(optimal_obj - curr_obj) / max(fabs(optimal_obj),
                                               fabs(curr_obj))
    return rew

  def _compute_reward(self, prev_obj, curr_obj, mip_stats):
    # old way of assigning reward -> change to incremental delta rewards.
    # rew = -1 * curr_obj / milp.optimal_objective
    assert self.config.delta_reward != self.config.primal_gap_reward
    if self.config.delta_reward:
      rew = (prev_obj - curr_obj) / self.config.obj_normalizer
    elif self.config.primal_gap_reward:
      rew = -1.0 * self._primal_gap(curr_obj)
    else:
      raise Exception('unspecified reward scheme.')
    return rew

  def step(self, action):
    if self._reset_next_step:
      return self.reset()

    self._n_steps += 1
    milp = self.milp
    curr_sol = self._curr_soln
    curr_obj = self._curr_obj
    var_names = self._var_names
    # update graph features based on the action.
    globals_ = self._globals
    variable_nodes = self._variable_nodes
    obj_nodes = self._objective_nodes
    # action is the next node to unfix.
    action = int(action)
    mask = variable_nodes[:, Env.VARIABLE_MASK_FIELD]
    # check if the previous step's mask was successfully applied
    assert mask[action], mask

    globals_[Env.GLOBAL_UNFIX_LEFT] -= 1
    self._unfixed_variables.add(var_names[action])

    fixed_assignment = {
        var: curr_sol[var]
        for var in var_names if var not in self._unfixed_variables
    }

    # process the unfixed variables at this step and
    # run lp or sub-mip according to the step type
    # populates curr_sol, local_search_case and curr_lp_sol fields before exiting
    ###################################################
    # {
    if globals_[Env.GLOBAL_UNFIX_LEFT] == 0:
      # run mip
      local_search_case = True
      mip = milp.mip.fix(fixed_assignment, relax_integral_constraints=False)
      ass, curr_obj, mip_stats = self._scip_solve(mip)
      curr_sol = ass
      # # add back the newly found solutions for the sub-mip.
      # # this updates the current solution to the new local one.
      # curr_sol.update(ass)
      milp.mip.validate_sol(curr_sol)
      # reset the current solution to the newly found one.
      self._curr_soln = curr_sol
      self._prev_obj = self._curr_obj
      self._curr_obj = curr_obj
      assert curr_obj <= self._prev_obj + 1e-4, (curr_obj, self._prev_obj,
                                                 os.getpid())
      # reset fixed variables.
      self._unfixed_variables = set([
          var for var in var_names
          if isinstance(milp.mip.varname2var[var], ContinuousVariable)
      ])
      # restock the limit for unfixes in this episode.
      globals_[Env.GLOBAL_UNFIX_LEFT] = self.k
      curr_lp_sol = curr_sol
      curr_lp_obj = curr_obj
      self._n_local_moves += 1
    else:
      # run lp
      local_search_case = False
      if self.config.lp_features:
        mip = milp.mip.fix(fixed_assignment, relax_integral_constraints=True)
        ass, curr_lp_obj, _ = self._scip_solve(mip)
        curr_lp_sol = ass
      else:
        curr_lp_sol = curr_sol
        curr_lp_obj = curr_obj
      # for var, val in fixed_assignment.items():
      #   # ass should only contain unfixed variables.
      #   assert var not in ass
      #   curr_lp_sol[var] = val

      # # add back the newly found variable assignments for the lp.
      # curr_lp_sol.update(ass)
    # }
    ###################################################

    ## Estimate reward
    if local_search_case:
      # lower the objective the better (minimization)
      if milp.is_optimal:
        assert curr_obj - milp.optimal_objective >= -1e-4, (
            curr_obj, milp.optimal_objective, os.getpid())
      rew = self._compute_reward(self._prev_obj, curr_obj, mip_stats)
      self._qualities.append(self._primal_gap(curr_obj))
      self._final_quality = self._qualities[-1]
      self._best_quality = min(self._qualities)
      self._mip_stats = mip_stats
      self._mip_works.append(mip_stats.mip_work)
    else:
      rew = 0
    self._ep_return += rew

    ## update the node features.
    variable_nodes = self._reset_mask(variable_nodes)
    if not local_search_case:
      for node in self._unfixed_variables:
        variable_nodes[self._varnames2varidx[node], Env.
                       VARIABLE_MASK_FIELD] = 0

    variable_nodes[:, Env.VARIABLE_CURR_ASSIGNMENT_FIELD] = [
        curr_sol[k] for k in var_names
    ]
    variable_nodes[:, Env.VARIABLE_LP_SOLN_FIELD] = [
        curr_lp_sol[k] for k in var_names
    ]
    variable_nodes[:, Env.VARIABLE_LP_SOLN_SLACK_UP_FIELD] = np_slack_up(
        variable_nodes[:, Env.VARIABLE_LP_SOLN_FIELD])
    variable_nodes[:, Env.VARIABLE_LP_SOLN_SLACK_DOWN_FIELD] = np_slack_down(
        variable_nodes[:, Env.VARIABLE_LP_SOLN_FIELD])

    obj_nodes[:, Env.
              OBJ_LP_VALUE_FIELD] = curr_lp_obj / self.config.obj_normalizer
    obj_nodes[:, Env.
              OBJ_INT_VALUE_FIELD] = curr_obj / self.config.obj_normalizer

    globals_[Env.GLOBAL_STEP_NUMBER] = self._n_steps / np.sqrt(
        self._steps_per_episode)
    globals_[Env.GLOBAL_N_LOCAL_MOVES] = self._n_local_moves
    globals_[Env.GLOBAL_LOCAL_SEARCH_STEP] = local_search_case

    self._globals = globals_
    self._variable_nodes = variable_nodes
    self._objective_nodes = obj_nodes

    if self._n_steps == self._steps_per_episode:
      self._reset_next_step = True
      self._prev_ep_return = self._ep_return
      self._prev_avg_quality = np.mean(self._qualities)
      self._prev_best_quality = self._best_quality
      self._prev_final_quality = self._final_quality
      self._prev_mean_work = np.mean(self._mip_works)
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

    return nest.map_structure_with_path(mk_spec, obs)

  def action_spec(self):
    return BoundedArraySpec((),
                            np.int32,
                            minimum=0,
                            maximum=len(self._variable_nodes) - 1,
                            name='action_spec')

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
