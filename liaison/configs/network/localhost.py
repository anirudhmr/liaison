"""All components on localhost. """

from liaison.utils import ConfigDict


def get_config():
  config = {
      'host_names':
      dict(localhost='127.0.0.1'),
      'hosts':
      dict(localhost=dict(
          use_ssh=False,
          base_dir='/home/ubuntu/ml4opt/liaison',
          shell_setup_commands=['cd /home/ubuntu/ml4opt/liaison/'],
          components=[
              'learner',
              'actor-*',
              'replay',
              'ps',
              'tensorplex',
              'loggerplex',
              'tensorboard',
              'systemboard',
              'irs',
          ]))
  }

  return ConfigDict(**config)
