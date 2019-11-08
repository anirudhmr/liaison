"""Bundle together shell config and actor config in one file here. """

from liaison.utils import ConfigDict


def get_config():
  config = ConfigDict()

  # required fields.
  config.class_path = "liaison.env.xor_env"  # should be rel to the parent directory.
  config.class_name = "Env"

  # makes observations suitable for the MLP model.
  config.make_obs_for_mlp = False

  # makes observations for graphnet agent with node labels in node features and
  # shortest path embedded as edge features.
  config.make_obs_for_graphnet_semi_supervised = False
  """if graph_seed < 0, then use the environment seed"""
  config.graph_seed = 42

  return config
