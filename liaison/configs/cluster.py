from liaison.utils import ConfigDict


def get_config():
  config = ConfigDict()
  config.host_names = dict(
      surreal_tmux='ec2-52-14-254-34.us-east-2.compute.amazonaws.com',
      surreal_tmux2='ec2-13-58-35-146.us-east-2.compute.amazonaws.com')

  config.host_info = dict(
      surreal_tmux=dict(base_dir='/home/ubuntu/ml4opt/liaison',
                        use_ssh=False,
                        shell_setup_commands=[],
                        spy_port=4007),
      surreal_tmux2=dict(base_dir='/home/ubuntu/ml4opt/liaison',
                         use_ssh=True,
                         ssh_username='ubuntu',
                         ssh_key_file='/home/ubuntu/.ssh/temp',
                         shell_setup_commands=['cd ${HOME}/nfs/liaison'],
                         spy_port=4007))

  assert sorted(config.host_names.keys()) == sorted(config.host_info.keys())
  return config
