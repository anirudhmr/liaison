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
  """if graph_seed < 0, then use the environment seed"""
  config.graph_seed = 42

  config.dataset = 'milp-facilities-3'
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
  config.primal_integral_reward = False

  return config
