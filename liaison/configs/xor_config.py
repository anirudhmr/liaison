from liaison.utils import ConfigDict


def get_config():
  config = ConfigDict()
  config.agent_config = ConfigDict()
  config.agent_config.network = ConfigDict()

  config.shell_config = ConfigDict()
  config.shell_config.use_gpu = False

  config.session_config = ConfigDict()
  config.session_config.sync_period = 100

  return config
