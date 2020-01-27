"""Bundle together shell config and actor config in one file here. """

from liaison.utils import ConfigDict


def get_config():
  config = ConfigDict()

  # required fields.
  config.class_path = "liaison.env.rins"  # should be rel to the parent directory.
  config.class_name = "Env"

  # makes observations suitable for the MLP model.
  config.make_obs_for_mlp = False
  # adds all the constraints to MLP state space.
  # adds #variables * #constraints dimensions to the state space.
  config.mlp_embed_constraints = False

  config.make_obs_for_self_attention = False
  config.make_obs_for_graphnet = True
  config.make_obs_for_bipartite_graphnet = False
  """if graph_seed < 0, then use the environment seed"""
  config.graph_seed = 42

  # specify dataset by dataset_path or dataset
  config.dataset_path = ''
  config.dataset = ''
  config.dataset_type = 'train'
  config.graph_start_idx = 0
  config.n_graphs = 1

  config.max_nodes = -1
  config.max_edges = -1

  config.k = 5
  config.n_local_moves = 100

  config.lp_features = True

  config.delta_reward = False
  config.primal_gap_reward = True

  config.disable_maxcuts = False
  config.enable_curriculum = False

  config.sample_size_schedule = ConfigDict()
  config.sample_size_schedule.start_value = 40
  config.sample_size_schedule.max_value = 100
  config.sample_size_schedule.start_step = 10000
  config.sample_size_schedule.dec_steps = 25000

  return config
