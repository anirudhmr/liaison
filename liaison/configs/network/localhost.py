"""All components on localhost. """

from liaison.utils import ConfigDict


def get_config():
  config = dict(
      localhost=dict(ssh_command='',
                     setup_commands=['cd /home/ubuntu/ml4opt/liaison/'],
                     components=[
                         'learner',
                         'actor-*',
                         'replay',
                         'ps',
                         'tensorplex',
                         'loggerplex',
                         'tensorboard',
                         'irs',
                     ]))

  return ConfigDict(**config)
