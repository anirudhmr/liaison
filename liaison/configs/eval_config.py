from liaison.utils import ConfigDict


def get_config():
  config = ConfigDict()

  config.n_evaluators = 3
  config.batch_size = 64

  config.use_parallel_envs = True
  config.use_threaded_envs = False

  config.dataset_type_field = 'dataset_type'

  return config
