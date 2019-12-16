from liaison.utils import ConfigDict


def get_config():
  config = ConfigDict()

  config.n_trials = 3
  config.eval_sleep_time = 10 * 60
  config.batch_size = 64

  config.use_parallel_envs = True
  config.use_threaded_envs = False

  # env accepts this field as kwargs.
  config.dataset_type_field = 'dataset_type'

  return config
