import liaison.utils as U
import numpy as np
from absl import app
from liaison.utils import ConfigDict
from tqdm import trange


def get_env_config():
  config = ConfigDict()

  # required fields.
  config.class_path = "liaison.env.rins_v2"  # should be rel to the parent directory.
  config.class_name = "Env"

  # makes observations suitable for the MLP model.
  config.make_obs_for_mlp = False
  # adds all the constraints to MLP state space.
  # adds #variables * #constraints dimensions to the state space.
  config.mlp_embed_constraints = False

  config.make_obs_for_self_attention = False
  config.make_obs_for_graphnet = False
  config.make_obs_for_bipartite_graphnet = True

  # specify dataset by dataset_path or dataset
  config.dataset_path = ''
  config.dataset = 'milp-cauction-100-filtered'
  config.dataset_type = 'train'
  config.graph_start_idx = 0
  config.n_graphs = 1000

  config.max_nodes = 800
  config.max_edges = 12000

  config.k = 5
  config.n_local_moves = 20

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
  config.dataset_schedule.datasets = [
      'milp-cauction-25-filtered', 'milp-cauction-100-filtered', 'milp-cauction-300-filtered'
  ]
  # should be len-1 where len is len of datasets.
  config.dataset_schedule.start_steps = [50000, 100000]

  # add one hot node labels for debugging graphnet models.
  config.attach_node_labels = False

  # multi dimensional action space.
  config.muldi_actions = True

  return config


def make_env():
  env_config = get_env_config()
  Env = U.import_obj(env_config.class_name, env_config.class_path)
  return Env(0, 42, **env_config)


def main(argv):
  env = make_env()
  ts = env.reset()

  for i in trange(1000):
    mask = ts.observation['mask']
    act = np.random.choice(len(mask), env.k, replace=False, p=mask / sum(mask))
    ts = env.step(act)


if __name__ == '__main__':
  app.run(main)
