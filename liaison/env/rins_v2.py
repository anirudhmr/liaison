# New and improved rins environment:
# Uses presolved model.
# Based directly on scip.Model without using additional primitives.
import copy
import os
import pickle

import numpy as np

import liaison.utils as U
from liaison.daper.dataset_constants import (DATASET_INFO_PATH, DATASET_PATH,
                                             LENGTH_MAP,
                                             NORMALIZATION_CONSTANTS)
from liaison.daper.milp.primitives import (ContinuousVariable, IntegerVariable,
                                           MIPInstance)
from liaison.daper.milp.scip_mip import SCIPMIPInstance
from liaison.daper.milp.scip_utils import del_scip_model
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
    self._optimal_soln = np.float32([milp.optimal_solution[v] for v in self._var_names])
    self._optimal_lp_soln = np.float32([milp.optimal_lp_sol[v] for v in self._var_names])

    globals_ = np.zeros(Env.N_GLOBAL_FIELDS, dtype=np.float32)
    globals_[Env.GLOBAL_STEP_NUMBER] = self._n_steps / np.sqrt(self.k * self.max_local_moves)
    globals_[Env.GLOBAL_UNFIX_LEFT] = self.k
    globals_[Env.GLOBAL_N_LOCAL_MOVES] = self._n_local_moves
    self._globals = globals_

  def _init_features(self, var_names, v_f):
    mip = self.mip
    # first construct variable nodes
    variable_nodes = np.zeros((len(var_names), Env.N_VARIABLE_FIELDS + v_f['values'].shape[1]),
                              dtype=np.float32)
    # mask all integral variables
    variable_nodes[:, Env.VARIABLE_IS_INTEGER_FIELD] = [
        v.vtype() in ['BINARY', 'INTEGER'] for v in mip.vars
    ]
    variable_nodes[:, Env.VARIABLE_OPTIMAL_LP_SOLN_FIELD] = [
        self.milp.optimal_lp_sol[v] for v in var_names
    ]
    if 'cont_variable_normalizer' in self.config:
      variable_nodes[:, Env.VARIABLE_OPTIMAL_LP_SOLN_FIELD] /= self.config.cont_variable_normalizer
    variable_nodes[:, Env.VARIABLE_OPTIMAL_LP_SOLN_SLACK_DOWN_FIELD] = np_slack_down(
        variable_nodes[:, Env.VARIABLE_OPTIMAL_LP_SOLN_FIELD])
    variable_nodes[:, Env.VARIABLE_OPTIMAL_LP_SOLN_SLACK_UP_FIELD] = np_slack_up(
        variable_nodes[:, Env.VARIABLE_OPTIMAL_LP_SOLN_FIELD])
    variable_nodes = self._reset_mask(variable_nodes)
    variable_nodes[:, Env.VARIABLE_OBJ_COEFF_FIELD] = list(
        map(lambda v: v.getObj() / self.config.obj_coeff_normalizer, mip.vars))
    assert list(map(lambda v: v.lstrip('t_'),
                    v_f['var_names'])) == self._var_names, (self._milp_choice, self._var_names,
                                                            v_f['var_names'])
    variable_nodes[:, Env.N_VARIABLE_FIELDS:] = v_f['values']
    objective_nodes = np.zeros((1, Env.N_OBJECTIVE_FIELDS), dtype=np.float32)
    # TODO: Add variable stats fields here.
    self._objective_nodes = objective_nodes
    self._variable_nodes = variable_nodes

  def _encode_static_bipartite_graph_features(self, e_f):
    milp = self.milp
    receivers, senders = e_f['indices']
    return dict(edges=pad_first_dim(np.float32(e_f['values']), self._max_edges),
                senders=pad_first_dim(senders, self._max_edges),
                receivers=pad_first_dim(receivers, self._max_edges),
                n_edge=np.int32(len(e_f['values'])))

  def _observation(self):
    obs = super(Env, self)._observation()
    obs['n_actions'] = np.int32(self.k)
    return obs

  def reset(self):
    # fix the graph for a few episodes to reduce load on the disk while loading datasets.
    if self._n_resets % self._sample_every_n_resets == 0:
      self.milp, self._sol, self._obj = self._sample()
      self.mip = SCIPMIPInstance.fromMIPInstance(self.milp.mip)
      # clean up previous scip model
      if hasattr(self, '_sub_mip_model'):
        del_scip_model(self._sub_mip_model)
      self._sub_mip_model = self.mip.get_scip_model()
    sol, obj, mip = self._sol, self._obj, self.mip
    c_f, e_f, v_f = load_pickled_features(self._dataset, self._dataset_type, self._milp_choice)
    self._var_names = var_names = list(map(lambda v: v.name.lstrip('t_'), mip.vars))
    self._init_ds()
    self._init_features(var_names, v_f)
    self._constraint_nodes = np.float32(c_f['values'])
    self._change_sol(sol, obj, sol, obj)
    self._best_quality = self._primal_gap(obj)
    self._final_quality = self._best_quality
    self._qualities = [self._best_quality]

    self._unfixed_variables = filter(lambda v: v.vtype() == 'CONTINUOUS', mip.vars)
    self._unfixed_variables = set(map(lambda v: v.name.lstrip('t_'), self._unfixed_variables))

    # static features computed only once per episode.
    if self.config.make_obs_for_graphnet:
      raise Exception('Not yet supported!')
      self._static_graph_features = self._encode_static_graph_features()
    elif self.config.make_obs_for_bipartite_graphnet:
      self._static_graph_features = self._encode_static_bipartite_graph_features(e_f)

    self._n_resets += 1
    return restart(self._observation())

  def _step_multidimensional_actions(self, action):
    if self._reset_next_step:
      return self.reset()

    assert len(action) == self.max_k
    action = action[:self.k]

    self._n_steps += 1
    milp = self.milp
    mip = self.mip
    curr_sol = self._curr_soln
    curr_obj = self._curr_obj
    var_names = self._var_names
    globals_ = self._globals
    variable_nodes = self._variable_nodes
    sub_mip_model = self._sub_mip_model
    mask = variable_nodes[:, Env.VARIABLE_MASK_FIELD]

    # check if action is valid.
    # check if the previous step's mask was successfully applied
    for act in action:
      assert mask[act], mask
    # no duplicates
    assert len(set(action)) == len(action)

    action_unfixes = set([var_names[i] for i in action])
    unfixed_vars = action_unfixes | self._unfixed_variables

    fixed_assignment = {var: curr_sol[var] for var in var_names if var not in unfixed_vars}
    assert len(fixed_assignment) == len(var_names) - len(unfixed_vars)

    # run sub-mip
    sub_mip_model.freeTransform()
    mip.fix(fixed_assignment, relax_integral_constraints=False, scip_model=sub_mip_model)
    ass, curr_obj, mip_stats = self._scip_solve(mip=None, solver=sub_mip_model)
    curr_sol = ass
    # # add back the newly found solutions for the sub-mip.
    # # this updates the current solution to the new local one.
    # reset the current solution to the newly found one.
    self._curr_soln = curr_sol
    self._prev_obj = self._curr_obj
    self._curr_obj = curr_obj
    assert curr_obj <= self._prev_obj + 1e-4, (curr_obj, self._prev_obj, os.getpid())

    # restock the limit for unfixes in this episode.
    curr_lp_sol = curr_sol
    curr_lp_obj = curr_obj
    self._n_local_moves += 1
    # lower the objective the better (minimization)
    if milp.is_optimal:
      assert curr_obj - milp.optimal_objective >= -1e-4, (curr_obj, milp.optimal_objective,
                                                          os.getpid())
    rew = self._compute_reward(self._prev_obj, curr_obj, mip_stats)
    self._qualities.append(self._primal_gap(curr_obj))
    self._final_quality = self._qualities[-1]
    self._best_quality = min(self._qualities)
    self._mip_stats = mip_stats
    self._mip_works.append(mip_stats.mip_work)
    self._ep_return += rew

    ## update the node features.
    variable_nodes = self._reset_mask(variable_nodes)
    for node in unfixed_vars:
      variable_nodes[self._varnames2varidx[node], Env.VARIABLE_MASK_FIELD] = 0

    # update the solution.
    self._change_sol(curr_sol, curr_obj, curr_lp_sol, curr_lp_obj)

    globals_[Env.GLOBAL_STEP_NUMBER] = self._n_steps / np.sqrt(self.k * self.max_local_moves)
    globals_[Env.GLOBAL_N_LOCAL_MOVES] = self._n_local_moves
    globals_[Env.GLOBAL_LOCAL_SEARCH_STEP] = True

    self._globals = globals_
    self._variable_nodes = variable_nodes

    if self._n_steps == self.k * self.max_local_moves:
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
      return super(Env, self).step(action)

  def action_spec(self):
    if self.config.muldi_actions:
      if self.config.k_schedule.enable:
        max_k = max(self.config.k_schedule.values)
      else:
        max_k = self.k
      return BoundedArraySpec((max_k, ),
                              np.int32,
                              minimum=0,
                              maximum=self._max_nodes - 1,
                              name='action_spec')
    else:
      return BoundedArraySpec((),
                              np.int32,
                              minimum=0,
                              maximum=self._max_nodes - 1,
                              name='action_spec')
