"""Bundle together shell config and actor config in one file here. """

from liaison.utils import ConfigDict


def get_config():
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
  config.dataset = ''
  config.dataset_type = 'train'
  config.graph_start_idx = 0
  config.n_graphs = 100000

  config.max_nodes = -1
  config.max_edges = -1

  config.k = 5
  config.n_local_moves = 100

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
  config.muldi_actions = False

  return config
