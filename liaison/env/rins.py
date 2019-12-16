import copy
import os
import pickle
from typing import Any, Dict, Text, Tuple, Union

import networkx as nx
import numpy as np
import scipy

import graph_nets as gn
from liaison.daper.dataset_constants import DATASET_PATH, LENGTH_MAP
from liaison.daper.milp.primitives import IntegerVariable, MIPInstance
from liaison.env import Env as BaseEnv
from liaison.env.environment import restart, termination, transition
from liaison.specs import ArraySpec, BoundedArraySpec
from liaison.utils import ConfigDict
from pyscipopt import Model, multidict, quicksum
from tensorflow.contrib.framework import nest


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

  N_VARIABLE_FIELDS = 4

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

  N_GLOBAL_FIELDS = 3

  def __init__(self,
               id,
               seed,
               graph_seed=-1,
               graph_start_idx=0,
               n_graphs=1,
               dataset='milp-facilities-3',
               dataset_type='train',
               k=5,
               steps_per_episode=60,
               **env_config):
    """If graph_seed < 0, then use the environment seed.
       k -> Max number of variables to unfix at a time.
            Informally, this is a bound on the local search
            neighbourhood size.
    """
    self.config = ConfigDict(env_config)
    self.id = id
    self.k = k
    self._steps_per_episode = steps_per_episode
    self.seed = seed
    self.set_seed(seed)
    if graph_seed < 0: graph_seed = seed
    self._setup_graph_random_state(graph_seed)

    self.milps = self._load_graph(dataset, dataset_type, graph_start_idx,
                                  n_graphs)

    # call reset so that obs_spec can work without calling reset
    self._ep_return = None
    self._prev_ep_return = 0
    self._prev_avg_ep_return = 0
    self._prev_best_ep_return = 0
    self._prev_final_ep_return = 0
    self._reset_next_step = True
    self.reset()

  def _load_graph(self, dataset: str, dataset_type: str, graph_start_idx: int,
                  n_graphs: int):
    path = DATASET_PATH[dataset]
    milps = []
    for graph_idx in range(graph_start_idx, graph_start_idx + n_graphs):
      with open(os.path.join(path, dataset_type, f'{graph_idx}.pkl'),
                'rb') as f:
        milp = pickle.load(f)
        milps.append(milp)
    return milps

  def _sample(self, choice=None):
    if choice is not None:
      return self.milps[choice]
    return np.random.choice(self.milps)

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
          constraint_features[cid, var_names.index(var_name)] = coeff

      features = np.hstack((features, constraint_features.flatten()))

    obs = dict(features=features, mask=mask)
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
               var_embeddings=var_embeddings)
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

    obs = dict(
        **obs,
        var_type_mask=self._var_type_mask,
        constraint_type_mask=self._constraint_type_mask,
        obj_type_mask=self._obj_type_mask,
        log_values=dict(
            ep_return=np.float32(self._prev_ep_return),
            avg_ep_return=np.float32(self._prev_avg_ep_return),
            best_ep_return=np.float32(self._prev_best_ep_return),
            final_ep_return=np.float32(self._prev_final_ep_return),
        ),
    )
    # masks should be mutually disjoint.
    assert not np.any(
        np.logical_and(obs['var_type_mask'], obs['constraint_type_mask']))
    assert not np.any(
        np.logical_and(obs['var_type_mask'], obs['obj_type_mask']))
    assert not np.any(
        np.logical_and(obs['constraint_type_mask'], obs['obj_type_mask']))
    assert np.all(
        np.logical_or(
            obs['var_type_mask'],
            np.logical_or(obs['constraint_type_mask'], obs['obj_type_mask'])))
    return obs

  def resample_and_reset(self, i):
    self.milp = self._sample(i)
    return self.reset(resample=False)

  def reset(self, resample=True):
    if resample:
      self.milp = self._sample()
    milp = self.milp
    self._ep_return = 0
    self._n_steps = 0
    self._n_local_moves = 0
    self._reset_next_step = False
    self._best_ep_return = milp.feasible_objective / milp.optimal_objective
    self._final_ep_return = milp.feasible_objective / milp.optimal_objective
    self._obj_vals = [milp.feasible_objective / milp.optimal_objective]
    self._var_names = var_names = list(milp.mip.varname2var.keys())
    # first construct variable nodes
    variable_nodes = np.zeros((len(var_names), Env.N_VARIABLE_FIELDS),
                              dtype=np.float32)
    # mask all variables
    variable_nodes[:, Env.VARIABLE_MASK_FIELD] = 1
    feas_sol = [milp.feasible_solution[v] for v in var_names]
    variable_nodes[:, Env.VARIABLE_CURR_ASSIGNMENT_FIELD] = feas_sol
    variable_nodes[:, Env.VARIABLE_LP_SOLN_FIELD] = feas_sol
    variable_nodes[:, Env.VARIABLE_IS_INTEGER_FIELD] = [
        isinstance(milp.mip.varname2var[var_name], IntegerVariable)
        for var_name in var_names
    ]

    # now construct constraint nodes
    constraint_nodes = np.zeros(
        (len(milp.mip.constraints), Env.N_CONSTRAINT_FIELDS), dtype=np.float32)
    for i, c in enumerate(milp.mip.constraints):
      c = c.cast_sense_to_le()
      constraint_nodes[i, Env.CONSTRAINT_CONSTANT_FIELD] = c.rhs / 5e3

    objective_nodes = np.zeros((1, Env.N_OBJECTIVE_FIELDS), dtype=np.float32)
    objective_nodes[:, 0] = milp.feasible_objective
    objective_nodes[:, 1] = milp.feasible_objective

    # get undirected edge representation first
    n_edges = sum([len(c) for c in milp.mip.constraints]) + len(milp.mip.obj)
    edges = np.zeros((n_edges, Env.N_EDGE_FIELDS), dtype=np.float32)
    senders = np.zeros((n_edges), dtype=np.int32)
    receivers = np.zeros((n_edges), dtype=np.int32)
    i = 0
    for cid, c in enumerate(milp.mip.constraints):
      c = c.cast_sense_to_le()
      for var_name, coeff in zip(c.expr.var_names, c.expr.coeffs):
        edges[i, Env.EDGE_WEIGHT_FIELD] = coeff
        # sender is the variable
        senders[i] = var_names.index(var_name)
        # receiver is the constraint.
        receivers[i] = len(variable_nodes) + cid
        i += 1

    for j, (var_name, coeff) in enumerate(
        zip(milp.mip.obj.expr.var_names, milp.mip.obj.expr.coeffs)):
      edges[i + j, Env.EDGE_WEIGHT_FIELD] = coeff
      senders[i + j] = var_names.index(var_name)
      receivers[i + j] = len(variable_nodes) + len(constraint_nodes)

    # now duplicate the edges to make them directed.
    edges = np.vstack((edges, edges))
    senders = np.hstack((senders, receivers))
    receivers = np.hstack((receivers, senders))

    globals_ = np.zeros(Env.N_GLOBAL_FIELDS, dtype=np.float32)
    globals_[Env.GLOBAL_STEP_NUMBER] = self._n_steps
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
    self._unfixed_variables = []
    self._curr_soln = copy.deepcopy(milp.feasible_solution)
    self._prev_obj = milp.feasible_objective
    self._curr_obj = milp.feasible_objective

    return restart(self._observation())

  def _scip_solve(self, mip: MIPInstance
                  ) -> Tuple[Dict[str, Union[int, float]], Union[int, float]]:
    """Solves a mip/lp using scip"""
    solver = Model()
    solver.hideOutput()
    mip.add_to_scip_solver(solver)
    solver.optimize()
    obj = float(solver.getObjVal())
    ass = {var.name: solver.getVal(var) for var in solver.getVars()}
    return ass, obj

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
    assert mask[action]

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
      mip = milp.mip.unfix(fixed_assignment, integral_relax=False)
      ass, curr_obj = self._scip_solve(mip)
      curr_sol = dict()
      for var, val in fixed_assignment.items():
        # ass should only contain unfixed variables.
        assert var not in ass
        curr_sol[var] = val

      # add back the newly found solutions for the sub-mip.
      # this updates the current solution to the new local one.
      curr_sol.update(ass)
      # reset the current solution to the newly found one.
      self._curr_soln = curr_sol
      self._prev_obj = self._curr_obj
      self._curr_obj = curr_obj
      # reset fixed variables.
      self._unfixed_variables = []
      # restock the limit for unfixes in this episode.
      globals_[Env.GLOBAL_UNFIX_LEFT] = self.k
      curr_lp_sol = curr_sol
      curr_lp_obj = curr_obj
      self._n_local_moves += 1
    else:
      # run lp
      local_search_case = False
      mip = milp.mip.unfix(fixed_assignment, integral_relax=True)
      ass, curr_lp_obj = self._scip_solve(mip)
      curr_lp_sol = dict()
      for var, val in fixed_assignment.items():
        # ass should only contain unfixed variables.
        assert var not in ass
        curr_lp_sol[var] = val

      # add back the newly found variable assignments for the lp.
      curr_lp_sol.update(ass)
    # }
    ###################################################

    ## Estimate reward
    if local_search_case:
      # lower the objective the better (minimization)
      if milp.is_optimal:
        assert curr_obj - milp.optimal_objective >= -1e-4, (
            curr_obj, milp.optimal_objective)
      # old way of assigning reward change to incremental delta rewards.
      # rew = -1 * curr_obj / milp.optimal_objective
      rew = (self._prev_obj - curr_obj) / milp.feasible_objective
      self._best_ep_return = min(curr_obj / milp.optimal_objective,
                                 self._best_ep_return)
      self._final_ep_return = curr_obj / milp.optimal_objective
      self._obj_vals.append(curr_obj / milp.optimal_objective)
    else:
      rew = 0
    self._ep_return += rew

    ## update the node features.
    if local_search_case:
      variable_nodes[:, Env.VARIABLE_MASK_FIELD] = 1
    else:
      variable_nodes[:, Env.VARIABLE_MASK_FIELD] = 1
      for node in self._unfixed_variables:
        variable_nodes[var_names.index(node), Env.VARIABLE_MASK_FIELD] = 0

    variable_nodes[:, Env.VARIABLE_CURR_ASSIGNMENT_FIELD] = [
        curr_sol[k] for k in var_names
    ]
    variable_nodes[:, Env.VARIABLE_LP_SOLN_FIELD] = [
        curr_lp_sol[k] for k in var_names
    ]
    obj_nodes[:, Env.
              OBJ_LP_VALUE_FIELD] = curr_lp_obj / milp.feasible_objective
    obj_nodes[:, Env.OBJ_INT_VALUE_FIELD] = curr_obj / milp.feasible_objective

    globals_[Env.GLOBAL_STEP_NUMBER] = self._n_steps / np.sqrt(
        self._steps_per_episode)
    globals_[Env.GLOBAL_N_LOCAL_MOVES] = self._n_local_moves

    if self._n_steps == self._steps_per_episode:
      self._reset_next_step = True
      self._prev_ep_return = self._ep_return
      self._prev_best_ep_return = self._best_ep_return
      self._prev_final_ep_return = self._final_ep_return
      self._prev_avg_ep_return = np.mean(self._obj_vals)
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
                            maximum=len(self._variable_nodes) - 1,
                            name='action_spec')

  def set_seed(self, seed):
    np.random.seed(seed + self.id)

  def _setup_graph_random_state(self, seed):
    self._graph_random_state = np.random.RandomState(seed=seed)
