from liaison.utils import ConfigDict


def get_config():
  config = ConfigDict()
  config.host_names = dict(surreal_tmux='127.0.0.1')

  config.host_info = dict(
      surreal_tmux=dict(base_dir='/home/ubuntu/ml4opt/liaison',
                        use_ssh=False,
                        shell_setup_commands=[],
                        spy_port=4007))

  assert sorted(config.host_names.keys()) == sorted(config.host_info.keys())
  return config
