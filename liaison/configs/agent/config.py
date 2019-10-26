from liaison.utils import ConfigDict
"""
Must have attribute class_path attribute.
"""


def get_config():
  config = ConfigDict()

  # required fields.
  config.class_path = "liaison.agents.ur_discrete"  # should be rel to the parent directory.
  config.class_name = "Agent"
  config.learning_rate = 1e-3

  config.model = ConfigDict()
  config.model.class_path = "liaison.agents.models.mlp"

  config.lr_init = 1e-3
  config.lr_min = 1e-4
  config.lr_start_decay_step = 1000
  config.lr_decay_steps = 1000
  config.lr_dec_val = .1
  config.lr_dec_approach = 'linear'

  config.ent_dec_init = 1
  config.ent_dec_min = 0
  config.ent_dec_steps = 1000
  config.ent_start_dec_step = 1000
  config.ent_dec_val = .1
  config.ent_dec_approach = 'linear'

  return config
