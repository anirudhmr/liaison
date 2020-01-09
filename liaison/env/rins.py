import copy
import os
import pickle
from math import fabs
from typing import Any, Dict, Text, Tuple, Union

import graph_nets as gn
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
from pyscipopt import Model


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
  # TODO: Add variable lower and upper bound vals.

  N_VARIABLE_FIELDS = 5

  # constant used in the constraints.
  CONSTRAINT_CONSTANT_FIELD = 0
  N_CONSTRAINT_FIELDS = 1

  # objective
  # current lp bound
  OBJ_LP_VALUE_FIELD = 0
  # current value achieved by
  # the integral solution.
  OBJ_INT_VALUE_FIELD = 1

  N_OBJECTIVE_FIELDS = 2

  # coefficient in the constraint as edge weight.
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
               dataset='milp-facilities-3',
               dataset_type='train',
               k=5,
               n_local_moves=10,
               max_nodes=-1,
               max_edges=-1,
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
    self.reset()

  def _sample(self, choice=None):
    n_graphs = self._n_graphs
    graph_start_idx = self._graph_start_idx

    if choice is None:
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
          constraint_features[cid, var_names.index(
              var_name)] = coeff / self.config.obj_coeff_normalizer

      features = np.hstack((features, constraint_features.flatten()))

    obs = dict(features=features, mask=mask, globals=self._globals)
    return obs

  def _observation_self_attention(self, nodes):

    mask = np.int32(self._variable_nodes[:, Env.VARIABLE_MASK_FIELD])
    features = np.hstack((nodes.flatten(), self._globals))
    cv = []
    for cid, c in enumerate(self.milp.mip.constraints):
      c = c.cast_sense_to_le()
      y = np.zeros(len(self._var_names), dtype=np.int32)
      for var_name, coeff in zip(c.expr.var_names, c.expr.coeffs):
        y[self._var_names.index(var_name)] = coeff
      cv.append(y)

    # var_embeddings[i][j] = c => ith variable uses jth constraint with coefficient c
    var_embeddings = np.transpose(np.asarray(cv, dtype=np.float32))

    obs = dict(mask=mask,
               var_nodes=np.float32(self._variable_nodes),
               var_embeddings=var_embeddings,
               globals=self._globals)
    return obs

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

  def _observation(self):
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

    obs = dict(**obs,
               var_type_mask=pad_last_dim(self._var_type_mask,
                                          self._max_nodes),
               constraint_type_mask=pad_last_dim(self._constraint_type_mask,
                                                 self._max_nodes),
               obj_type_mask=pad_last_dim(self._obj_type_mask,
                                          self._max_nodes),
               log_values=dict(
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
                   mip_work=np.float32(self._mip_work),
               ))
    # optimal solution can be used for supervised auxiliary tasks.
    obs['optimal_solution'] = pad_last_dim(
        np.float32([self.milp.optimal_solution[v] for v in self._var_names]),
        self._max_nodes)
    return obs

  def reset(self):
    milp = self.milp = self._sample()
    self._ep_return = 0
    self._n_steps = 0
    self._n_local_moves = 0
    self._reset_next_step = False
    self._best_quality = self._primal_gap(milp.feasible_objective)
    self._final_quality = self._best_quality
    self._qualities = [self._best_quality]
    self._mip_work = 0
    self._mip_works = []

    self._var_names = var_names = list(milp.mip.varname2var.keys())
    # first construct variable nodes
    variable_nodes = np.zeros((len(var_names), Env.N_VARIABLE_FIELDS),
                              dtype=np.float32)
    # mask all integral variables
    feas_sol = [milp.feasible_solution[v] for v in var_names]
    variable_nodes[:, Env.VARIABLE_CURR_ASSIGNMENT_FIELD] = feas_sol
    variable_nodes[:, Env.VARIABLE_LP_SOLN_FIELD] = feas_sol
    variable_nodes[:, Env.VARIABLE_IS_INTEGER_FIELD] = [
        isinstance(milp.mip.varname2var[var_name], IntegerVariable)
        for var_name in var_names
    ]
    variable_nodes[:, Env.VARIABLE_OPTIMAL_LP_SOLN_FIELD] = [
        milp.optimal_lp_sol[v] for v in var_names
    ]
    variable_nodes = self._reset_mask(variable_nodes)

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
    globals_[Env.GLOBAL_STEP_NUMBER] = self._n_steps / np.sqrt(
        self._steps_per_episode)
    globals_[Env.GLOBAL_UNFIX_LEFT] = self.k
    globals_[Env.GLOBAL_N_LOCAL_MOVES] = self._n_local_moves

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
    self._unfixed_variables = [
        var for var in var_names
        if isinstance(milp.mip.varname2var[var], ContinuousVariable)
    ]
    self._curr_soln = copy.deepcopy(milp.feasible_solution)
    self._curr_obj = milp.feasible_objective
    self._prev_obj = milp.feasible_objective
    return restart(self._observation())

  def _scip_solve(self, mip: MIPInstance):
    """Solves a mip/lp using scip"""
    solver = Model()
    solver.hideOutput()
    mip.add_to_scip_solver(solver)
    solver.optimize()
    assert solver.getStatus() == 'optimal', solver.getStatus()
    obj = float(solver.getObjVal())
    ass = {var.name: solver.getVal(var) for var in solver.getVars()}
    return ass, obj, solver.getNNodes()

  def _reset_mask(self, variable_nodes):
    variable_nodes[:, Env.
                   VARIABLE_MASK_FIELD] = variable_nodes[:, Env.
                                                         VARIABLE_IS_INTEGER_FIELD]
    return variable_nodes

  def _primal_gap(self, curr_obj):
    optimal_obj = self.milp.optimal_objective
    if curr_obj == 0 and optimal_obj == 0:
      rew = 0.
    elif np.sign(curr_obj) * np.sign(optimal_obj) < 0:
      rew = 1.
    else:
      rew = fabs(optimal_obj - curr_obj) / max(fabs(optimal_obj),
                                               fabs(curr_obj))
    return rew

  def _compute_reward(self, prev_obj, curr_obj, n_nodes):
    # old way of assigning reward -> change to incremental delta rewards.
    # rew = -1 * curr_obj / milp.optimal_objective
    assert self.config.delta_reward != self.config.primal_gap_reward
    if self.config.delta_reward:
      rew = (prev_obj - curr_obj) / self.config.obj_normalizer
    elif self.config.primal_gap_reward:
      rew = -1.0 * self._primal_gap(curr_obj)
    elif self.config.primal_integral_reward:
      raise Exception('Unspecified reward scheme.')
    else:
      raise Exception('Unspecified reward scheme.')
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
    self._unfixed_variables.append(var_names[action])

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
      ass, curr_obj, self._mip_work = self._scip_solve(mip)
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
      self._unfixed_variables = [
          var for var in var_names
          if isinstance(milp.mip.varname2var[var], ContinuousVariable)
      ]
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
      rew = self._compute_reward(self._prev_obj, curr_obj, self._mip_work)
      self._qualities.append(self._primal_gap(curr_obj))
      self._final_quality = self._qualities[-1]
      self._best_quality = min(self._qualities)
      self._mip_works.append(self._mip_work)
    else:
      rew = 0
    self._ep_return += rew

    ## update the node features.
    variable_nodes = self._reset_mask(variable_nodes)
    if not local_search_case:
      for node in self._unfixed_variables:
        variable_nodes[var_names.index(node), Env.VARIABLE_MASK_FIELD] = 0

    variable_nodes[:, Env.VARIABLE_CURR_ASSIGNMENT_FIELD] = [
        curr_sol[k] for k in var_names
    ]
    variable_nodes[:, Env.VARIABLE_LP_SOLN_FIELD] = [
        curr_lp_sol[k] for k in var_names
    ]
    obj_nodes[:, Env.
              OBJ_LP_VALUE_FIELD] = curr_lp_obj / self.config.obj_normalizer
    obj_nodes[:, Env.
              OBJ_INT_VALUE_FIELD] = curr_obj / self.config.obj_normalizer

    globals_[Env.GLOBAL_STEP_NUMBER] = self._n_steps / np.sqrt(
        self._steps_per_episode)
    globals_[Env.GLOBAL_N_LOCAL_MOVES] = self._n_local_moves
    globals_[Env.GLOBAL_LOCAL_SEARCH_STEP] = local_search_case

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
