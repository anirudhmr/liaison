# New and improved rins environment:
# Uses presolved model.
# Based directly on scip.Model without using additional primitives.
import copy
import os

import liaison.utils as U
import numpy as np
from liaison.daper.dataset_constants import (DATASET_PATH, LENGTH_MAP,
                                             NORMALIZATION_CONSTANTS)
from liaison.daper.milp.primitives import (ContinuousVariable, IntegerVariable,
                                           MIPInstance)
from liaison.daper.milp.scip_mip import SCIPMIPInstance
from liaison.env.environment import restart, termination, transition
from liaison.env.rins import Env as RINSEnv
from liaison.env.utils.rins import *
from liaison.specs import BoundedArraySpec
from liaison.utils import ConfigDict


class Env(RINSEnv):

  def _init_ds(self):
    # Initialize data structures
    milp = self.milp
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

    self._varnames2varidx = {}
    for i, var_name in enumerate(self._var_names):
      self._varnames2varidx[var_name] = i
    # optimal solution can be used for supervised auxiliary tasks.
    self._optimal_soln = np.float32(
        [milp.optimal_solution[v] for v in self._var_names])
    self._optimal_lp_soln = np.float32(
        [milp.optimal_lp_sol[v] for v in self._var_names])

    globals_ = np.zeros(Env.N_GLOBAL_FIELDS, dtype=np.float32)
    globals_[Env.GLOBAL_STEP_NUMBER] = self._n_steps / np.sqrt(
        self._steps_per_episode)
    globals_[Env.GLOBAL_UNFIX_LEFT] = self.k
    globals_[Env.GLOBAL_N_LOCAL_MOVES] = self._n_local_moves
    self._globals = globals_

  def _init_variable_features(self, var_names, v_f):
    mip = self.mip
    # first construct variable nodes
    variable_nodes = np.zeros(
        (len(var_names), Env.N_VARIABLE_FIELDS + v_f['values'].shape[1]),
        dtype=np.float32)
    # mask all integral variables
    feas_sol = [self.milp.feasible_solution[v] for v in var_names]
    variable_nodes[:, Env.VARIABLE_CURR_ASSIGNMENT_FIELD] = feas_sol
    variable_nodes[:, Env.VARIABLE_LP_SOLN_FIELD] = feas_sol
    variable_nodes[:, Env.VARIABLE_LP_SOLN_SLACK_UP_FIELD] = np_slack_up(
        feas_sol)
    variable_nodes[:, Env.VARIABLE_LP_SOLN_SLACK_DOWN_FIELD] = np_slack_down(
        feas_sol)
    variable_nodes[:, Env.VARIABLE_IS_INTEGER_FIELD] = [
        v.vtype() in ['BINARY', 'INTEGER'] for v in mip.vars
    ]
    variable_nodes[:, Env.VARIABLE_OPTIMAL_LP_SOLN_FIELD] = [
        self.milp.optimal_lp_sol[v] for v in var_names
    ]
    variable_nodes[:, Env.
                   VARIABLE_OPTIMAL_LP_SOLN_SLACK_DOWN_FIELD] = np_slack_down(
                       variable_nodes[:, Env.VARIABLE_OPTIMAL_LP_SOLN_FIELD])
    variable_nodes[:, Env.
                   VARIABLE_OPTIMAL_LP_SOLN_SLACK_UP_FIELD] = np_slack_up(
                       variable_nodes[:, Env.VARIABLE_OPTIMAL_LP_SOLN_FIELD])
    variable_nodes = self._reset_mask(variable_nodes)
    variable_nodes[:, Env.VARIABLE_OBJ_COEFF_FIELD] = list(
        map(lambda v: v.getObj() / self.config.obj_coeff_normalizer, mip.vars))
    assert list(map(lambda v: v.lstrip('t_'),
                    v_f['var_names'])) == self._var_names, (self._milp_choice,
                                                            self._var_names,
                                                            v_f['var_names'])
    variable_nodes[:, Env.N_VARIABLE_FIELDS:] = v_f['values']
    # TODO: Add variable stats fields here.
    self._variable_nodes = variable_nodes

  def _init_objective_nodes(self):
    objective_nodes = np.zeros((1, Env.N_OBJECTIVE_FIELDS), dtype=np.float32)
    objective_nodes[:,
                    0] = self.milp.feasible_objective / self.config.obj_normalizer
    objective_nodes[:,
                    1] = self.milp.feasible_objective / self.config.obj_normalizer
    self._objective_nodes = objective_nodes

  def _encode_static_bipartite_graph_features(self, e_f):
    milp = self.milp
    receivers, senders = e_f['indices']
    return dict(edges=pad_first_dim(np.float32(e_f['values']),
                                    self._max_edges),
                senders=pad_first_dim(senders, self._max_edges),
                receivers=pad_first_dim(receivers, self._max_edges),
                n_edge=np.int32(len(e_f['values'])))

  def reset(self):
    milp = self.milp = self._sample()
    mip = self.mip = SCIPMIPInstance.fromMIPInstance(milp.mip)
    c_f, e_f, v_f = mip.get_features()
    self._var_names = var_names = list(
        map(lambda v: v.name.lstrip('t_'), mip.vars))
    self._init_ds()
    self._init_variable_features(var_names, v_f)
    self._init_objective_nodes()
    self._constraint_nodes = np.float32(c_f['values'])

    self._unfixed_variables = set(
        filter(lambda v: v.vtype() == 'CONTINUOUS', mip.vars))
    self._curr_soln = copy.deepcopy(milp.feasible_solution)
    self._curr_obj = milp.feasible_objective
    self._prev_obj = milp.feasible_objective

    # static features computed only once per episode.
    if self.config.make_obs_for_graphnet:
      raise Exception('Not yet supported!')
      self._static_graph_features = self._encode_static_graph_features()
    elif self.config.make_obs_for_bipartite_graphnet:
      self._static_graph_features = self._encode_static_bipartite_graph_features(
          e_f)
    return restart(self._observation())

  def _update_lp_solution_fields(self, variable_nodes, curr_lp_sol):
    variable_nodes[:, Env.VARIABLE_LP_SOLN_FIELD] = [
        curr_lp_sol[k] for k in self._var_names
    ]
    variable_nodes[:, Env.VARIABLE_LP_SOLN_SLACK_UP_FIELD] = np_slack_up(
        variable_nodes[:, Env.VARIABLE_LP_SOLN_FIELD])
    variable_nodes[:, Env.VARIABLE_LP_SOLN_SLACK_DOWN_FIELD] = np_slack_down(
        variable_nodes[:, Env.VARIABLE_LP_SOLN_FIELD])
    return variable_nodes

  def _step(self, action):
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
      # reset unfixed variables.
      self._unfixed_variables = set([
          var for var in var_names
          if isinstance(milp.mip.varname2var[var], ContinuousVariable)
      ])
      # restock the limit for unfixes in this episode.
      globals_[Env.GLOBAL_UNFIX_LEFT] = self.k
      curr_lp_sol = curr_sol
      curr_lp_obj = curr_obj
      self._n_local_moves += 1
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
      # run lp
      local_search_case = False
      if self.config.lp_features:
        mip = milp.mip.fix(fixed_assignment, relax_integral_constraints=True)
        ass, curr_lp_obj, _ = self._scip_solve(mip)
        curr_lp_sol = ass
      else:
        curr_lp_sol = curr_sol
        curr_lp_obj = curr_obj
      rew = 0.

    ###################################################
    self._ep_return += rew

    ## update the node features.
    variable_nodes = self._reset_mask(variable_nodes)
    for node in self._unfixed_variables:
      variable_nodes[self._varnames2varidx[node], Env.VARIABLE_MASK_FIELD] = 0

    variable_nodes[:, Env.VARIABLE_CURR_ASSIGNMENT_FIELD] = [
        curr_sol[k] for k in var_names
    ]
    variable_nodes = self._update_lp_solution_fields(variable_nodes,
                                                     curr_lp_sol)

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

  def _step_multidimensional_actions(self, action):
    if self._reset_next_step:
      return self.reset()

    assert len(action) == self.k

    self._n_steps += 1
    milp = self.milp
    curr_sol = self._curr_soln
    curr_obj = self._curr_obj
    var_names = self._var_names
    globals_ = self._globals
    variable_nodes = self._variable_nodes
    obj_nodes = self._objective_nodes
    mask = variable_nodes[:, Env.VARIABLE_MASK_FIELD]
    # check if the previous step's mask was successfully applied
    for act in action:
      assert mask[act], mask

    globals_[Env.GLOBAL_UNFIX_LEFT] -= 1

    fixed_assignment = {
        var: curr_sol[var]
        for var in var_names if var not in set(action)
    }

    # run sub-mip
    mip = milp.mip.fix(fixed_assignment, relax_integral_constraints=False)
    ass, curr_obj, mip_stats = self._scip_solve(mip)
    curr_sol = ass
    # # add back the newly found solutions for the sub-mip.
    # # this updates the current solution to the new local one.
    milp.mip.validate_sol(curr_sol)
    # reset the current solution to the newly found one.
    self._curr_soln = curr_sol
    self._prev_obj = self._curr_obj
    self._curr_obj = curr_obj
    assert curr_obj <= self._prev_obj + 1e-4, (curr_obj, self._prev_obj,
                                               os.getpid())

    # restock the limit for unfixes in this episode.
    globals_[Env.GLOBAL_UNFIX_LEFT] = self.k
    curr_lp_sol = curr_sol
    curr_lp_obj = curr_obj
    self._n_local_moves += 1
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
    self._ep_return += rew

    ## update the node features.
    variable_nodes = self._reset_mask(variable_nodes)
    for node in self._unfixed_variables:
      variable_nodes[self._varnames2varidx[node], Env.VARIABLE_MASK_FIELD] = 0

    variable_nodes[:, Env.VARIABLE_CURR_ASSIGNMENT_FIELD] = [
        curr_sol[k] for k in var_names
    ]
    variable_nodes = self._update_lp_solution_fields(variable_nodes,
                                                     curr_lp_sol)

    obj_nodes[:, Env.
              OBJ_LP_VALUE_FIELD] = curr_lp_obj / self.config.obj_normalizer
    obj_nodes[:, Env.
              OBJ_INT_VALUE_FIELD] = curr_obj / self.config.obj_normalizer

    globals_[Env.GLOBAL_STEP_NUMBER] = self._n_steps / np.sqrt(
        self._steps_per_episode)
    globals_[Env.GLOBAL_N_LOCAL_MOVES] = self._n_local_moves
    globals_[Env.GLOBAL_LOCAL_SEARCH_STEP] = True

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

  def step(self, action):
    if self.config.muldi_actions:
      return self._step_multidimensional_actions(action)
    else:
      return self._step(action)

  def action_spec(self):
    if self.config.muldi_actions:
      return BoundedArraySpec((self.k, ),
                              np.int32,
                              minimum=0,
                              maximum=len(self._variable_nodes) - 1,
                              name='action_spec')
    else:
      return BoundedArraySpec((),
                              np.int32,
                              minimum=0,
                              maximum=len(self._variable_nodes) - 1,
                              name='action_spec')
