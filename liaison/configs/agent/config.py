from liaison.utils import ConfigDict


"""
Must have attribute class_path attribute.
"""


def get_config():
  config = ConfigDict()

  config.model = ConfigDict()

  config.lr_init = 1e-4
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

  # specify <= 0 here to disable grad clip
  config.grad_clip = 1.0
  config.discount_factor = 0.99

  return config
