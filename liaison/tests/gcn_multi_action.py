import argparse
import os
import pdb

import numpy as np
from tqdm import tqdm

import liaison.utils as U
import tensorflow as tf
from absl import flags, logging
from absl.testing import absltest
from liaison.agents import StepOutput
from liaison.agents.gcn import Agent as GCNAgent
from liaison.distributed import Trajectory
from liaison.distributed.learner_for_test import Learner
from liaison.distributed.shell_for_test import Shell
from liaison.env.batch import SerialBatchedEnv
# use this to get large graph.
from liaison.env.utils.shortest_path import generate_networkx_graph
from liaison.specs.specs import BoundedArraySpec
from liaison.utils import ConfigDict
from tensorflow.contrib.framework import nest
from tensorflow.python.client import timeline

FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'gcn', 'Options: mlp, gcn')
flags.DEFINE_boolean('enable_gpu', True, 'Enables gpu for tf')
flags.DEFINE_integer('B', 8, 'batch_size')
flags.DEFINE_integer('T', 9, 'batch_size')
flags.DEFINE_boolean('testStep', False, 'Test step')
flags.DEFINE_boolean('testUpdate', False, 'Test update')

# set after parsing flags
SEED = 42
B = None
T = None


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
  config.dataset = 'milp-cauction-100-filtered'
  config.dataset_type = 'train'
  config.graph_start_idx = 0
  config.n_graphs = 1

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
  config.model.num_blocks = 1
  config.model.d_ff = 16
  config.model.num_heads = 1
  config.model.d_model = 32
  config.model.dropout_rate = 0.
  config.model.use_mlp_value_func = False

  # The following code duplicated in gcn_rins.py as well.
  # Propagate any changes made as needed.
  config.model.model_kwargs = ConfigDict()
  config.model.model_kwargs.class_path = "liaison.agents.models.bipartite_gcn_rins"
  config.model.model_kwargs.n_prop_layers = 1
  config.model.model_kwargs.edge_embed_dim = 16
  config.model.model_kwargs.node_embed_dim = 16
  config.model.model_kwargs.global_embed_dim = 16
  config.model.model_kwargs.node_hidden_layer_sizes = [16]
  config.model.model_kwargs.edge_hidden_layer_sizes = [16]
  config.model.model_kwargs.policy_torso_hidden_layer_sizes = [16, 16]
  config.model.model_kwargs.value_torso_hidden_layer_sizes = [16, 16]
  config.model.model_kwargs.policy_summarize_hidden_layer_sizes = [16]
  config.model.model_kwargs.value_summarize_hidden_layer_sizes = [16]
  config.model.model_kwargs.supervised_prediction_torso_hidden_layer_sizes = [16, 16]
  config.model.model_kwargs.sum_aggregation = False
  config.model.model_kwargs.use_layer_norm = True

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
  return config


def _get_env():
  env_config = get_env_config()
  env_class = U.import_obj(env_config.class_name, env_config.class_path)
  return SerialBatchedEnv(B, env_class, [env_config] * B, SEED)


def _create_shell(env):
  agent_config = get_agent_config()
  agent_class = U.import_obj(agent_config.class_name, agent_config.class_path)
  action_spec = env.action_spec()
  obs_spec = env.observation_spec()
  return Shell(action_spec,
               obs_spec,
               seed=SEED,
               agent_class=agent_class,
               agent_config=agent_config,
               batch_size=B,
               use_gpu=FLAGS.enable_gpu)


def _create_learner(env, shell):
  agent_config = get_agent_config()
  agent_class = U.import_obj(agent_config.class_name, agent_config.class_path)
  traj_spec = Trajectory(env.observation_spec(), step_output_spec=shell.step_output_spec()).spec
  traj_spec = Trajectory.format_traj_spec(traj_spec, B, T)
  action_spec = env.action_spec()
  action_spec.set_shape((B, ) + action_spec.shape[1:])
  return Learner(seed=SEED,
                 traj_spec=traj_spec,
                 action_spec=action_spec,
                 agent_class=agent_class,
                 agent_config=dict(vis_loggers=[], **agent_config),
                 use_gpu=FLAGS.enable_gpu)


def get_session():
  if FLAGS.enable_gpu:
    return tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))
  return tf.Session()


class VtraceAgentTest(absltest.TestCase):

  def _setup(self):
    global B, T
    B = FLAGS.B
    T = FLAGS.T

  def testStep(self):
    if not FLAGS.testStep:
      return
    self._setup()
    env = _get_env()
    shell = _create_shell(env)

    ts = env.reset()
    for i in tqdm(range(50)):
      step_output = shell.step(step_type=ts.step_type,
                               reward=ts.reward,
                               observation=ts.observation)
      ts = env.step(step_output.action)
    print('Test complete!')

  def _sample_trajectory(self, env, shell):
    global B, T
    ts = env.reset()
    traj = Trajectory(obs_spec=env.observation_spec(), step_output_spec=shell.step_output_spec())
    traj.reset()
    traj.start(next_state=shell.next_state, **dict(ts._asdict()))
    for i in range(T):
      step_output = shell.step(step_type=ts.step_type,
                               reward=ts.reward,
                               observation=ts.observation)
      ts = env.step(step_output.action)
      traj.add(step_output=step_output, **dict(ts._asdict()))
    exps = traj.debatch_and_stack()
    return exps

  def testUpdate(self):
    global B, T
    if not FLAGS.testUpdate:
      return
    self._setup()
    env = _get_env()
    shell = _create_shell(env)
    learner = _create_learner(env, shell)
    exps = self._sample_trajectory(env, shell)
    batch = learner.batch_and_preprocess_trajs(exps)
    print('***************')
    print('Starting....')
    print('***************')

    for i in tqdm(range(100)):
      learner.update(batch)

    print('Test complete!')


if __name__ == '__main__':
  absltest.main()
