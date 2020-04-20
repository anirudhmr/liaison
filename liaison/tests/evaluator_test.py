import argparse
import os
import pdb
import unittest

import liaison.utils as U
import numpy as np
import tensorflow as tf
from absl import flags, logging
from absl.testing import absltest
from liaison.agents import StepOutput
from liaison.agents.gcn import Agent as GCNAgent
from liaison.distributed import Trajectory
from liaison.distributed.evaluators.evaluator import Evaluator
from liaison.distributed.learner_for_test import Learner
from liaison.distributed.shell_for_test import Shell
from liaison.env.batch import SerialBatchedEnv
# use this to get large graph.
from liaison.env.utils.shortest_path import generate_networkx_graph
from liaison.loggers import AvgPipeLogger, ConsoleLogger, DownSampleLogger
from liaison.specs.specs import BoundedArraySpec
from liaison.utils import ConfigDict
from tensorflow.contrib.framework import nest
from tensorflow.python.client import timeline
from tqdm import tqdm, trange

# set after parsing flags
SEED = 42
B = 4
T = 9


def get_env_config():
  """Bundle together shell config and actor config in one file here. """
  config = ConfigDict()

  # required fields.
  config.class_path = "liaison.env.rins_v2"  # should be rel to the parent directory.
  config.class_name = "Env"

  # makes observations suitable for the MLP model.
  config.make_obs_for_mlp = False
  # adds all the constraints to MLP state space.
  # adds #variables * #constraints dimensions to the state space.
  config.mlp_embed_constraints = False

  config.make_obs_for_graphnet = False
  config.make_obs_for_bipartite_graphnet = True

  # specify dataset by dataset_path or dataset
  config.dataset_path = ''
  config.dataset = 'milp-cauction-300-filtered'
  config.dataset_type = 'train'
  config.graph_start_idx = 0
  config.n_graphs = 100

  config.max_nodes = 3000
  config.max_edges = 25000

  config.k = 5
  config.n_local_moves = 10

  config.lp_features = False

  config.delta_reward = False
  config.primal_gap_reward = True
  config.disable_maxcuts = False

  # starting solution hamming distance schedule
  config.starting_sol_schedule = ConfigDict()
  config.starting_sol_schedule.enable = False
  config.starting_sol_schedule.start_value = 1
  config.starting_sol_schedule.max_value = 100
  config.starting_sol_schedule.start_step = 10000
  config.starting_sol_schedule.dec_steps = 25000

  # dataset schedule
  config.dataset_schedule = ConfigDict()
  config.dataset_schedule.enable = False
  config.dataset_schedule.datasets = ['milp-cauction-100-filtered', 'milp-cauction-300-filtered']
  config.dataset_schedule.start_steps = [50000]  # should be len-1 where len is len of datasets.

  config.k_schedule = ConfigDict()
  config.k_schedule.enable = False
  config.k_schedule.values = [5, 10]
  config.k_schedule.start_steps = [50000]

  config.n_local_move_schedule = ConfigDict()
  # if enabled config.n_local_moves will be disabled
  config.n_local_move_schedule.enable = False
  config.n_local_move_schedule.start_step = 10000
  config.n_local_move_schedule.start_value = 5
  config.n_local_move_schedule.max_value = 25
  config.n_local_move_schedule.dec_steps = 25000

  # add one hot node labels for debugging graphnet models.
  config.attach_node_labels = False

  # multi dimensional action space.
  config.muldi_actions = True
  return config


def get_agent_config():
  config = ConfigDict()

  config.lr_init = 1e-4
  config.lr_min = 1e-7
  config.lr_start_dec_step = 1000
  config.lr_dec_steps = 1000
  config.lr_dec_val = .1
  config.lr_dec_approach = 'linear'

  config.ent_dec_init = 1e-2
  config.ent_dec_min = 0.0
  config.ent_dec_steps = 1000
  config.ent_start_dec_step = 1000
  # dec_val not used for linear scheme
  config.ent_dec_val = .1
  config.ent_dec_approach = 'linear'

  # specify <= 0 here to disable grad clip
  config.grad_clip = 1.0
  config.discount_factor = 1.0

  config.optimizer = ConfigDict()
  # Options: Adam or RMSProp.
  config.optimizer.name = 'Adam'
  # hyperparams for RMSProp
  config.optimizer.decay = .9
  config.optimizer.momentum = 0.0
  config.optimizer.epsilon = 1e-7
  # required fields.
  config.class_path = "liaison.agents.gcn_multi_actions"
  config.class_name = "Agent"

  config.model = ConfigDict()
  config.model.class_path = 'liaison.agents.models.transformer_auto_regressive'
  config.model.num_blocks = 4
  config.model.d_ff = 32
  config.model.num_heads = 4
  config.model.d_model = 64
  config.model.dropout_rate = 0.
  config.model.use_mlp_value_func = False

  # The following code duplicated in gcn_rins.py as well.
  # Propagate any changes made as needed.
  config.model.model_kwargs = ConfigDict()
  config.model.model_kwargs.class_path = "liaison.agents.models.bipartite_gcn_rins"
  config.model.model_kwargs.n_prop_layers = 4
  config.model.model_kwargs.edge_embed_dim = 16
  config.model.model_kwargs.node_embed_dim = 32
  config.model.model_kwargs.global_embed_dim = 32
  config.model.model_kwargs.node_hidden_layer_sizes = [16, 16]
  config.model.model_kwargs.edge_hidden_layer_sizes = [16, 16]
  config.model.model_kwargs.policy_torso_hidden_layer_sizes = [16, 16]
  config.model.model_kwargs.value_torso_hidden_layer_sizes = [16, 16]
  config.model.model_kwargs.policy_summarize_hidden_layer_sizes = [16]
  config.model.model_kwargs.value_summarize_hidden_layer_sizes = [16]
  config.model.model_kwargs.supervised_prediction_torso_hidden_layer_sizes = [16, 16]
  config.model.model_kwargs.sum_aggregation = False
  config.model.model_kwargs.use_layer_norm = True
  config.model.model_kwargs.apply_gradient_to_graphnet_every = 1
  config.model.model_kwargs.memory_hack = False

  config.clip_rho_threshold = 1.0
  config.clip_pg_rho_threshold = 1.0

  config.loss = ConfigDict()
  config.loss.vf_loss_coeff = 1.0

  config.loss.al_coeff = ConfigDict()
  config.loss.al_coeff.init_val = 0.
  config.loss.al_coeff.min_val = 0.
  config.loss.al_coeff.start_decay_step = int(1e10)
  config.loss.al_coeff.decay_steps = 5000
  # dec_val not used for linear scheme
  config.loss.al_coeff.dec_val = .1
  config.loss.al_coeff.dec_approach = 'linear'

  # applicable for agent 'liaison.agents.gcn_large_batch'
  config.apply_grads_every = 1
  config.log_features_every = 0

  config.freeze_graphnet_weights_step = 50 + 10

  return config


def get_shell_config():
  config = ConfigDict()
  agent_config = get_agent_config()
  # shell class path is default to the distributed folder.
  config.class_path = 'liaison.distributed.shell_for_test'
  config.class_name = 'Shell'
  config.agent_scope = 'shell'
  config.use_gpu = True
  config.agent_class = U.import_obj(agent_config.class_name, agent_config.class_path)
  config.agent_config = agent_config
  config.agent_config.update(evaluation_mode=True)
  return config


def get_eval_config():
  config = ConfigDict()

  config.n_trials = 1
  config.eval_sleep_time = 0
  config.batch_size = 4

  config.use_parallel_envs = False
  config.use_threaded_envs = True

  # env accepts this field as kwargs.
  config.dataset_type_field = 'dataset_type'
  config.env_config = ConfigDict()
  config.env_config.n_local_moves = 10
  config.env_config.lp_features = False
  config.env_config.delta_reward = False
  config.env_config.primal_gap_reward = True
  config.env_config.n_graphs = 1

  config.starting_sol_schedule = ConfigDict(enable=False)
  config.dataset_schedule = ConfigDict(enable=False)
  config.k_schedule = ConfigDict(enable=False)
  config.n_local_move_schedule = ConfigDict(enable=False)

  return config


def get_env_class():
  env_config = get_env_config()
  return U.import_obj(env_config.class_name, env_config.class_path)


def get_shell_class():
  shell_config = get_shell_config()
  return U.import_obj(shell_config.class_name, shell_config.class_path)


def get_env_configs():
  # create env configs
  eval_config = get_eval_config()

  env_configs = []
  for i in range(eval_config.batch_size):
    c = get_env_config()
    c.update({
        eval_config.dataset_type_field: 'train',
        'graph_start_idx': i,
        **eval_config.env_config
    })
    env_configs.append(c)
  return env_configs


def setup_evaluator_loggers(evaluator_name):
  loggers = []
  loggers.append(AvgPipeLogger(ConsoleLogger(print_every=1, name=evaluator_name)))
  return loggers


def create_evaluator():
  return Evaluator(shell_class=get_shell_class(),
                   shell_config=get_shell_config(),
                   env_class=get_env_class(),
                   env_configs=get_env_configs(),
                   loggers=setup_evaluator_loggers('evaluator'),
                   heuristic_loggers=setup_evaluator_loggers(f'heuristic'),
                   seed=SEED,
                   **get_eval_config())


class EvaluatorTest(unittest.TestCase):

  def testHeuristic(self):
    evaluator = create_evaluator()
    print('***************')
    print('Starting....')
    print('***************')

    t = evaluator.get_heuristic_thread()
    t.start()
    t.join()

    print('==============')
    print('Done!')
    print('==============')

  # def testLoop(self):
  #   evaluator = create_evaluator()
  #   print('***************')
  #   print('Starting....')
  #   print('***************')

  #   for _ in trange(5):
  #     evaluator.run_loop(1)

  #   print('==============')
  #   print('Done!')
  #   print('==============')


if __name__ == '__main__':
  absltest.main()
