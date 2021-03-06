import copy
import math
import os
import pdb
import pickle
from math import fabs
from typing import Any, Dict, Text, Tuple, Union

import graph_nets as gn
import liaison.utils as U
import networkx as nx
import numpy as np
import scipy
import tree as nest
from liaison.daper.dataset_constants import LENGTH_MAP, NORMALIZATION_CONSTANTS
from liaison.daper.milp.primitives import IntegerVariable, MIPInstance
from liaison.env import Env as BaseEnv
from liaison.env.environment import restart, termination, transition
from liaison.env.utils.rins import *
from liaison.env.utils.rins import get_sample
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
  # add a flag to indicate if the variable appears in the objective or not.
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

  # At which step was this variable unfixed.
  # Normalized to be in [0, 1]
  VARIABLE_UNFIX_STEP = 15

  N_VARIABLE_FIELDS = 16

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
               graph_start_idx=0,
               n_graphs=1,
               dataset='',
               dataset_type='train',
               k=5,
               n_local_moves=10,
               max_nodes=-1,
               max_edges=-1,
               sample_every_n_resets=1,
               **env_config):
    """k -> Max number of variables to unfix at a time.
            Informally, this is a bound on the local search
            neighbourhood size.
       max_nodes, max_edges -> Use for padding
    """
    self.config = ConfigDict(env_config)
    self.id = id
    self.k = self._original_k = k
    if self.config.k_schedule.enable:
      self.max_k = max(self.config.k_schedule.values)
    else:
      self.max_k = k
    self.max_local_moves = n_local_moves
    self.seed = seed
    if max_nodes < 0:
      max_nodes = NORMALIZATION_CONSTANTS[dataset]['max_nodes']
    if max_edges < 0:
      max_edges = NORMALIZATION_CONSTANTS[dataset]['max_edges']
    self._max_nodes = max_nodes
    self._max_edges = max_edges
    self.set_seed(seed)
    self._dataset = dataset
    self._dataset_type = dataset_type
    self._max_graphs = n_graphs
    self._graph_start_idx = graph_start_idx
    self._sample_every_n_resets = sample_every_n_resets

    if dataset:
      self.config.update(NORMALIZATION_CONSTANTS[dataset])
    # call reset so that obs_spec can work without calling reset
    self._ep_return = None
    self._prev_ep_return = np.nan
    self._prev_avg_quality = np.nan
    self._prev_best_quality = np.nan
    self._prev_final_quality = np.nan
    self._prev_mean_work = np.nan
    self._prev_k = np.nan
    self._reset_next_step = True
    if 'SYMPH_PS_SERVING_HOST' in os.environ:
      self._global_step_fetcher = GlobalStepFetcher(min_request_spacing=4)
    else:
      self._global_step_fetcher = None
    # map from sample to length of the mip
    self._sample_lengths = None
    self._n_resets = 0
    self._vars_unfixed_so_far = []
    self.reset()

  def _get_n_graphs(self):
    return min(self._max_graphs, LENGTH_MAP[self._dataset][self._dataset_type])

  def _pick_sample_for_curriculum(self, choice):
    config = self.config
    graph_start_idx = self._graph_start_idx

    if config.starting_sol_schedule.enable or config.dataset_schedule.enable or config.k_schedule.enable or config.n_local_move_schedule.enable:
      if self._global_step_fetcher is None:
        raise Exception('step fetcher not found!')
      step = self._global_step_fetcher.get()
    else:
      return None

    # Schedule for the dataset
    if config.dataset_schedule.enable:
      self._dataset = config.dataset_schedule.datasets[0]
      for i, s in enumerate(map(int, config.dataset_schedule.start_steps)):
        if step >= s:
          self._dataset = config.dataset_schedule.datasets[i + 1]
      self.config.update(NORMALIZATION_CONSTANTS[self._dataset])

    if choice is not None:
      sample_idx = choice
    else:
      sample_idx = self._rnd_state.choice(
          list(range(graph_start_idx, graph_start_idx + self._get_n_graphs())))

    starting_sol = None
    starting_obj = None
    # Schedule for starting solution's hamming distance
    if config.starting_sol_schedule.enable:
      hamming_dist = linear_interpolate_inc(step, config.starting_sol_schedule.start_step,
                                            config.starting_sol_schedule.dec_steps,
                                            config.starting_sol_schedule.start_value,
                                            config.starting_sol_schedule.max_value)

      hamming_dist_to_sol = get_hamming_dists(self._dataset, self._dataset_type, sample_idx)
      assert len(hamming_dist_to_sol) > 0
      for d in sorted(hamming_dist_to_sol.keys()):
        starting_sol, starting_obj = hamming_dist_to_sol[d]
        if d >= hamming_dist:
          break

    # Schedule for k
    if config.k_schedule.enable:
      self.k = config.k_schedule.values[0]
      for i, s in enumerate(config.k_schedule.start_steps):
        if step >= s:
          self.k = config.k_schedule.values[i + 1]

    # Schedule for the # local moves
    if config.n_local_move_schedule.enable:
      self.max_local_moves = int(
          linear_interpolate_inc(
              step,
              config.n_local_move_schedule.start_step,
              config.n_local_move_schedule.dec_steps,
              config.n_local_move_schedule.start_value,
              config.n_local_move_schedule.max_value,
          ))
    return sample_idx, starting_sol, starting_obj

  def _sample(self, choice=None):
    graph_start_idx = self._graph_start_idx
    status = self._pick_sample_for_curriculum(choice)
    if status is not None:
      choice, sol, obj = status
    elif choice is None:
      choice = self._rnd_state.choice(
          range(graph_start_idx, graph_start_idx + self._get_n_graphs()))
      sol, obj = None, None

    self._milp_choice = choice
    milp = get_sample(self._dataset, self._dataset_type, choice)
    if sol is None:
      return milp, milp.feasible_solution, milp.feasible_objective
    else:
      return milp, sol, obj

  def _observation_mlp(self, nodes):
    mask = np.int32(self._variable_nodes[:, Env.VARIABLE_MASK_FIELD])
    features = np.hstack((self._variable_nodes.flatten(), self._globals))

    if self.config.mlp_embed_constraints:
      milp = self.milp
      var_names = self._var_names
      # constraint_features[i, j] = coefficient for the
      #                             jth variable in the ith constraint
      constraint_features = np.zeros((len(milp.mip.constraints), len(var_names)), dtype=np.float32)

      for cid, c in enumerate(milp.mip.constraints):
        c = c.cast_sense_to_le()
        for var_name, coeff in zip(c.expr.var_names, c.expr.coeffs):
          constraint_features[
              cid, self._varnames2varidx[var_name]] = coeff / self.config.obj_coeff_normalizer

      features = np.hstack((features, constraint_features.flatten()))

    obs = dict(features=np.float32(features),
               mask=mask,
               mlp_mask=mask,
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
    node_mask[0:len(self._variable_nodes)] = self._variable_nodes[:, Env.VARIABLE_MASK_FIELD]

    node_mask = pad_last_dim(node_mask, self._max_nodes)
    graph_features = self._pad_graph_features(graph_features)
    obs = dict(graph_features=graph_features,
               node_mask=node_mask,
               mask=node_mask,
               **self._static_graph_features[1])
    return obs

  def _observation_bipartite_graphnet(self):
    graph_features = dict(left_nodes=pad_first_dim(self._variable_nodes, self._max_nodes),
                          right_nodes=pad_first_dim(self._constraint_nodes, self._max_nodes),
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

    mask = self._variable_nodes[:, Env.VARIABLE_MASK_FIELD]
    if self.config.adapt_k.enable:
      mask = pad_last_dim(mask, 1 + self._max_nodes)
      mask[-1] = self._stop_switch_mask
    else:
      mask = pad_last_dim(mask, self._max_nodes)
    return dict(graph_features=graph_features, node_mask=mask, mask=mask)

  def _observation(self):
    variable_nodes = self._variable_nodes
    constraint_nodes = self._constraint_nodes
    objective_nodes = self._objective_nodes

    nodes = np.zeros(
        (len(variable_nodes) + len(constraint_nodes) + len(objective_nodes),
         variable_nodes.shape[1] + constraint_nodes.shape[1] + objective_nodes.shape[1]),
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

    if self.config.make_obs_for_graphnet:
      obs.update(self._observation_graphnet_inductive(nodes))

    if self.config.make_obs_for_bipartite_graphnet:
      obs.update(self._observation_bipartite_graphnet())

    obs = dict(
        **obs,
        max_k=np.int32(self.max_k),
        n_local_moves=self._globals[Env.GLOBAL_N_LOCAL_MOVES],
        optimal_solution=pad_last_dim(self._optimal_soln, self._max_nodes),
        optimal_lp_solution=pad_last_dim(self._optimal_lp_soln, self._max_nodes),
        current_solution=pad_last_dim(np.float32([self._curr_soln[v] for v in self._var_names]),
                                      self._max_nodes),
        log_values=dict(  # useful for tensorboard.
            ep_return=np.float32(self._prev_ep_return),
            avg_quality=np.float32(self._prev_avg_quality),
            best_quality=np.float32(self._prev_best_quality),
            final_quality=np.float32(self._prev_final_quality),
            mip_work=np.float32(self._prev_mean_work),
            k_val=np.float32(self._prev_k),
        ),
        curr_episode_log_values=dict(ep_return=np.float32(self._ep_return),
                                     avg_quality=np.float32(np.mean(self._qualities)),
                                     best_quality=np.float32(self._best_quality),
                                     final_quality=np.float32(self._final_quality),
                                     curr_obj=np.float32(self._curr_obj),
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
        edges[i, Env.EDGE_WEIGHT_FIELD] = coeff / self.config.constraint_coeff_normalizer
        # sender is the variable
        senders[i] = self._varnames2varidx[var_name]
        # receiver is the constraint.
        receivers[i] = len(self._variable_nodes) + cid
        i += 1

    for j, (var_name,
            coeff) in enumerate(zip(milp.mip.obj.expr.var_names, milp.mip.obj.expr.coeffs)):
      edges[i + j, Env.EDGE_WEIGHT_FIELD] = coeff / self.config.obj_coeff_normalizer
      senders[i + j] = self._varnames2varidx[var_name]
      receivers[i + j] = len(self._variable_nodes) + len(self._constraint_nodes)

    # now duplicate the edges to make them directed.
    edges = np.vstack((edges, edges))
    senders, receivers = np.hstack((senders, receivers)), np.hstack((receivers, senders))

    # compute masks for each node type
    variable_nodes = self._variable_nodes
    objective_nodes = self._objective_nodes
    constraint_nodes = self._constraint_nodes
    n_nodes = len(variable_nodes) + len(objective_nodes) + len(constraint_nodes)
    var_type_mask = np.zeros((n_nodes), dtype=np.int32)
    var_type_mask[0:len(variable_nodes)] = 1
    constraint_type_mask = np.zeros((n_nodes), dtype=np.int32)
    constraint_type_mask[len(variable_nodes):len(variable_nodes) + len(constraint_nodes)] = 1
    obj_type_mask = np.zeros((n_nodes), dtype=np.int32)
    obj_type_mask[len(variable_nodes) + len(constraint_nodes):len(variable_nodes) +
                  len(constraint_nodes) + len(objective_nodes)] = 1

    # masks should be mutually disjoint.
    assert not np.any(np.logical_and(var_type_mask, constraint_type_mask))
    assert not np.any(np.logical_and(var_type_mask, obj_type_mask))
    assert not np.any(np.logical_and(constraint_type_mask, obj_type_mask))
    assert np.all(np.logical_or(var_type_mask, np.logical_or(constraint_type_mask, obj_type_mask)))

    return dict(edges=edges, senders=senders, receivers=receivers, n_edge=np.int32(
        len(edges))), dict(var_type_mask=pad_last_dim(var_type_mask, self._max_nodes),
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
        edges[i, Env.EDGE_WEIGHT_FIELD] = coeff / self.config.constraint_coeff_normalizer
        # sender is the variable
        senders[i] = self._varnames2varidx[var_name]
        # receiver is the constraint.
        receivers[i] = cid
        i += 1
    return dict(edges=pad_first_dim(edges, self._max_edges),
                senders=pad_first_dim(senders, self._max_edges),
                receivers=pad_first_dim(receivers, self._max_edges),
                n_edge=np.int32(len(edges)))

  def _change_sol(self, sol, obj_val, lp_sol, lp_obj_val):
    # initialize or change the current solution
    var_names = self._var_names
    milp = self.milp
    variable_nodes = self._variable_nodes

    # mask all integral variables
    feas_sol = [sol[v] for v in var_names]
    feas_lp_sol = [lp_sol[v] for v in var_names]

    variable_nodes[:, Env.VARIABLE_CURR_ASSIGNMENT_FIELD] = feas_sol
    variable_nodes[:, Env.VARIABLE_LP_SOLN_FIELD] = feas_lp_sol
    if 'cont_variable_normalizer' in self.config:
      variable_nodes[:, Env.VARIABLE_CURR_ASSIGNMENT_FIELD] /= self.config.cont_variable_normalizer
      variable_nodes[:, Env.VARIABLE_LP_SOLN_FIELD] /= self.config.cont_variable_normalizer

    variable_nodes[:, Env.VARIABLE_LP_SOLN_SLACK_UP_FIELD] = np_slack_up(feas_lp_sol)
    variable_nodes[:, Env.VARIABLE_LP_SOLN_SLACK_DOWN_FIELD] = np_slack_down(feas_lp_sol)
    self._variable_nodes = variable_nodes
    self._objective_nodes[:, 0] = obj_val / self.config.obj_normalizer
    self._objective_nodes[:, 1] = lp_obj_val / self.config.obj_normalizer

    self._curr_soln = copy.deepcopy(sol)
    if hasattr(self, '_curr_obj'):
      self._prev_obj = self._curr_obj
    self._curr_obj = obj_val

  def reset_solution(self, sol, obj_val):
    # call at the beginning of a local move.
    # assert self._globals[Env.GLOBAL_UNFIX_LEFT] == self.k
    self._change_sol(sol, obj_val, sol, obj_val)

    self._best_quality = self._primal_gap(obj_val)
    self._final_quality = self._best_quality
    self._qualities = [self._best_quality]
    return restart(self._observation())

  def _scip_solve(self, solver):
    """solves a mip/lp using scip"""
    if solver is None:
      solver = Model()
    solver.hideOutput()
    if self.config.disable_maxcuts:
      for param in [
          'separating/maxcuts', 'separating/maxcutsroot', 'propagating/maxrounds',
          'propagating/maxroundsroot', 'presolving/maxroundsroot'
      ]:
        solver.setIntParam(param, 0)

      solver.setBoolParam('conflict/enable', False)
      solver.setPresolve(SCIP_PARAMSETTING.OFF)

    solver.setBoolParam('randomization/permutevars', True)
    # seed is set to 0 permanently.
    solver.setIntParam('randomization/permutationseed', 0)
    solver.setIntParam('randomization/randomseedshift', 0)

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
    return ass, obj, mip_stats

  def _reset_mask(self, variable_nodes):
    variable_nodes[:, Env.VARIABLE_MASK_FIELD] = variable_nodes[:, Env.VARIABLE_IS_INTEGER_FIELD]
    return variable_nodes

  def _primal_gap(self, curr_obj):
    optimal_obj = self.milp.optimal_objective
    if curr_obj == 0 and optimal_obj == 0:
      rew = 1.
    elif np.sign(curr_obj) * np.sign(optimal_obj) < 0:
      rew = 1.
    else:
      rew = fabs(optimal_obj - curr_obj) / max(fabs(optimal_obj), fabs(curr_obj))
    return rew

  def _compute_reward(self, prev_obj, curr_obj, mip_stats):
    # old way of assigning reward -> change to incremental delta rewards.
    # rew = -1 * curr_obj / milp.optimal_objective
    if self.config.delta_reward:
      rew = (prev_obj - curr_obj) / self.config.obj_normalizer
    elif self.config.primal_gap_reward:
      rew = -1.0 * self._primal_gap(curr_obj)
    elif self.config.primal_gap_reward_with_work:
      rew = -1.0 * self._primal_gap(curr_obj) * (1 +
                                                 mip_stats.mip_work / self.config.work_normalizer)
    else:
      raise Exception('unspecified reward scheme.')
    return rew

  def _set_stop_switch_mask(self):
    self._stop_switch_mask = (self._n_steps_in_this_local_move >= self.config.adapt_k.min_k)

  def _add_rens_submip_bounds(self, sub_mip_model):
    milp = self.milp
    optimal_lp_sol = milp.optimal_lp_sol
    varname2var = {}
    for var in sub_mip_model.getVars(transformed=False):
      varname2var[var.name.lstrip('t_')] = var
    for vname in unfix_vars:
      # change the bounds of unfixed integer variables.
      if isinstance(milp.mip.varname2var[vname], IntegerVariable):
        var = varname2var[vname]
        f, c = math.floor(optimal_lp_sol[vname]), math.ceil(optimal_lp_sol[vname])
        if c == f:
          continue
        sub_mip_model.chgVarLbGlobal(var, f)
        sub_mip_model.chgVarUbGlobal(var, c)

  def _add_sol_to_submip(self, model, sol):
    # convert sol to sol_scip
    sol_scip = model.createSol()
    varname2var = {v.name.lstrip('t_'): v for v in model.getVars()}
    for var_name, val in sol.items():
      try:
        model.setSolVal(sol_scip, varname2var[var_name], val)
      except Exception as e:
        pass
    model.addSol(sol_scip)

  def step(self, action):
    if self._reset_next_step:
      return self.reset()
    self._n_steps += 1
    self._n_steps_in_this_local_move += 1
    milp = self.milp
    mip = self.mip
    curr_sol = self._curr_soln
    curr_obj = self._curr_obj
    var_names = self._var_names
    globals_ = self._globals
    variable_nodes = self._variable_nodes
    sub_mip_model = self._sub_mip_model
    vars_unfixed_so_far = self._vars_unfixed_so_far
    mask = variable_nodes[:, Env.VARIABLE_MASK_FIELD]
    # action is the next node to unfix.
    action = int(action)
    # check if the previous step's mask was successfully applied
    globals_[Env.GLOBAL_UNFIX_LEFT] -= 1
    if action < self._max_nodes:
      assert mask[action], mask
      local_search_case = (globals_[Env.GLOBAL_UNFIX_LEFT] == 0)
      vars_unfixed_so_far.append(var_names[action])
    else:
      assert self.config.adapt_k.enable
      local_search_case = True
    unfix_vars = self._unfixed_variables | set(vars_unfixed_so_far)
    fixed_assignment = {var: curr_sol[var] for var in var_names if var not in unfix_vars}

    # process the unfixed variables at this step and
    # run lp or sub-mip according to the step type
    # populates curr_sol, local_search_case and curr_lp_sol fields before exiting
    ###################################################
    # {
    if local_search_case:
      # run mip
      sub_mip_model.freeTransform()
      mip.fix(fixed_assignment, relax_integral_constraints=False, scip_model=sub_mip_model)
      if self.config.use_rens_submip_bounds:
        self._add_rens_submip_bounds(sub_mip_model)
      self._add_sol_to_submip(sub_mip_model, curr_sol)
      ass, curr_obj, mip_stats = self._scip_solve(sub_mip_model)
      curr_sol = ass
      # # add back the newly found solutions for the sub-mip.
      # # this updates the current solution to the new local one.
      # curr_sol.update(ass)
      milp.mip.validate_sol(curr_sol)
      # reset the current solution to the newly found one.
      self._curr_soln = curr_sol
      self._prev_obj = self._curr_obj
      self._curr_obj = curr_obj
      # assert curr_obj <= self._prev_obj + 1e-4, (curr_obj, self._prev_obj, os.getpid())
      # restock the limit for unfixes in this episode.
      globals_[Env.GLOBAL_UNFIX_LEFT] = self.k
      curr_lp_sol = curr_sol
      curr_lp_obj = curr_obj
      self._n_local_moves += 1
      self._n_steps_in_this_local_move = 0
      self._vars_unfixed_so_far = []
    else:
      # run lp
      if self.config.lp_features:
        sub_mip_model.freeTransform()
        mip = mip.fix(fixed_assignment, relax_integral_constraints=True, scip_model=sub_mip_model)
        ass, curr_lp_obj, _ = self._scip_solve(sub_mip_model)
        curr_lp_sol = ass
      else:
        curr_lp_sol = curr_sol
        curr_lp_obj = curr_obj
    # }
    ###################################################

    ## Estimate reward
    if local_search_case:
      # lower the objective the better (minimization)
      # if milp.is_optimal:
      #   assert curr_obj - milp.optimal_objective >= -1e-4, (curr_obj, milp.optimal_objective)
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
    if local_search_case:
      variable_nodes[:, Env.VARIABLE_UNFIX_STEP] = 0
    else:
      variable_nodes[:, Env.VARIABLE_UNFIX_STEP] *= (self._n_steps_in_this_local_move - 1)
      variable_nodes[action, Env.VARIABLE_UNFIX_STEP] = self._n_steps_in_this_local_move
      variable_nodes[:, Env.VARIABLE_UNFIX_STEP] /= self._n_steps_in_this_local_move
      for node in unfix_vars:
        variable_nodes[self._varnames2varidx[node], Env.VARIABLE_MASK_FIELD] = 0

    # update the solution.
    self._change_sol(curr_sol, curr_obj, curr_lp_sol, curr_lp_obj)

    globals_[Env.GLOBAL_STEP_NUMBER] = self._n_steps / np.sqrt(self.k * self.max_local_moves)
    globals_[Env.GLOBAL_N_LOCAL_MOVES] = self._n_local_moves
    globals_[Env.GLOBAL_LOCAL_SEARCH_STEP] = local_search_case
    self._globals = globals_
    self._variable_nodes = variable_nodes
    self._set_stop_switch_mask()

    self._reset_next_step |= self._n_local_moves >= self.max_local_moves
    if self._reset_next_step:
      self._prev_ep_return = self._ep_return
      self._prev_avg_quality = np.mean(self._qualities)
      self._prev_best_quality = self._best_quality
      self._prev_final_quality = self._final_quality
      self._prev_mean_work = np.mean(self._mip_works)
      self._prev_k = self._n_steps / self._n_local_moves
      return termination(rew, self._observation())
    else:
      return transition(rew, self._observation())

  def observation_spec(self):
    # Note: this function will be called before reset call is issued and
    # observation might not be ready.
    # But for this specific environment it works.
    obs = self._observation()

    def mk_spec(path_tuple, np_arr):
      return ArraySpec(np_arr.shape, np_arr.dtype, name='_'.join(path_tuple) + '_spec')

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
                    receivers=pad_first_dim(features.receivers, self._max_edges))
    return dict(**features)

  def set_seed(self, seed):
    np.random.seed(seed + self.id)
    self._rnd_state = np.random.RandomState(seed=seed + self.id)

  def varname2idx(self, var_name):
    return self._var_names.index(var_name)

  def get_varnames(self):
    return list(self._var_names)

  def get_curr_soln(self):
    sol = dict(self._curr_soln)
    # remove variables from sol which are pruned by pre-solving step.
    # for k in set(sol.keys()) - set(self._var_names):
    #   del sol[k]
    return sol
