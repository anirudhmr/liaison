"""All components on localhost. """

from liaison.utils import ConfigDict


def get_config():
  config = dict(
      localhost=dict(use_ssh=False,
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

  return ConfigDict(**config)
