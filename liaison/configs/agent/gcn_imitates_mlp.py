from liaison.configs.agent.config import get_config as get_base_config
from liaison.configs.agent.mlp import get_config as get_mlp_config
from liaison.utils import ConfigDict


def get_config():
  config = get_base_config()

  # required fields.
  config.class_path = "liaison.agents.gcn_imitates_mlp"
  config.class_name = "Agent"

  config.model = ConfigDict()
  config.model.class_path = "liaison.agents.models.gcn_rins"
  config.model.n_prop_layers = 8
  config.model.node_hidden_layer_sizes = [32]
  config.model.edge_hidden_layer_sizes = [32]
  config.model.sum_aggregation = False

  config.mlp_model = get_mlp_config().model
  return config
