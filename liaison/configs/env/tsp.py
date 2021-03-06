"""Bundle together shell config and actor config in one file here. """

from liaison.utils import ConfigDict


def get_config():
  config = ConfigDict()

  # required fields.
  config.class_path = "liaison.env.tsp"  # should be rel to the parent directory.
  config.class_name = "Env"

  # makes observations suitable for the MLP model.
  config.make_obs_for_mlp = False
  """if graph_seed < 0, then use the environment seed"""
  config.graph_seed = 42

  config.dataset = 'tsp-20'
  config.dataset_type = 'train'
  config.graph_idx = 0

  return config
