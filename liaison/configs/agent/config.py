from liaison.utils import ConfigDict
"""
Must have attribute class_path attribute.
"""


def get_config():
  config = ConfigDict()

  # required fields.
  config.class_path = "liaison.agents.vtrace"
  config.class_name = "Agent"

  config.model = ConfigDict()
  config.model.class_path = "liaison.agents.models.mlp"
  config.model.hidden_layer_sizes = [32, 32]
  config.model.n_actions = 2

  config.lr_init = 1e-3
  config.lr_min = 1e-4
  config.lr_start_dec_step = 1000
  config.lr_dec_steps = 1000
  config.lr_dec_val = .1
  config.lr_dec_approach = 'linear'

  config.ent_dec_init = 1
  config.ent_dec_min = 0
  config.ent_dec_steps = 1000
  config.ent_start_dec_step = 1000
  config.ent_dec_val = .1
  config.ent_dec_approach = 'linear'

  config.grad_clip = 1.0
  config.discount_factor = 0.99
  config.clip_rho_threshold = 1.0
  config.clip_pg_rho_threshold = 1.0

  config.loss = ConfigDict()
  config.loss.vf_loss_coeff = 1.0

  return config
