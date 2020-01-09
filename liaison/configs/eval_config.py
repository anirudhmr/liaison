from liaison.utils import ConfigDict


def get_config():
  config = ConfigDict()

  config.n_trials = 1
  config.eval_sleep_time = 30 * 60  # (in seconds)
  config.batch_size = 32

  config.use_parallel_envs = True
  config.use_threaded_envs = False

  # env accepts this field as kwargs.
  config.dataset_type_field = 'dataset_type'
  config.env_config = ConfigDict()
  config.env_config.n_local_moves = 10
  config.env_config.lp_features = True

  return config
