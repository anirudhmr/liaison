from liaison.configs.agent.config import get_config as get_base_config
from liaison.utils import ConfigDict


def get_config():
  config = get_base_config()

  # required fields.
  config.class_path = "liaison.agents.mlp"
  config.class_name = "Agent"

  config.model = ConfigDict()
  config.model.class_path = "liaison.agents.models.mlp"
  config.model.hidden_layer_sizes = [32, 32]

  config.loss = ConfigDict()
  config.loss.vf_loss_coeff = 1.0

  return config
