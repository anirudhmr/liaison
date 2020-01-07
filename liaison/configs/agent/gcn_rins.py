from liaison.configs.agent.config import get_config as get_base_config
from liaison.utils import ConfigDict


def get_config():
  config = get_base_config()

  # required fields.
  config.class_path = "liaison.agents.gcn"
  config.class_name = "Agent"

  config.model = ConfigDict()
  config.model.class_path = "liaison.agents.models.gcn_rins"
  config.model.n_prop_layers = 4
  config.model.node_hidden_layer_sizes = [32]
  config.model.edge_hidden_layer_sizes = [32]
  config.model.sum_aggregation = True

  config.clip_rho_threshold = 1.0
  config.clip_pg_rho_threshold = 1.0

  config.loss = ConfigDict()
  config.loss.vf_loss_coeff = 1.0

  config.loss.al_coeff = ConfigDict()
  config.loss.al_coeff.init_val = 1.
  config.loss.al_coeff.min_val = 0.
  config.loss.al_coeff.start_decay_step = int(1e10)
  config.loss.al_coeff.decay_steps = 5000
  # dec_val not used for linear scheme
  config.loss.al_coeff.dec_val = .1
  config.loss.al_coeff.dec_approach = 'linear'

  return config
