from liaison.utils import ConfigDict
"""If prefix of the process name matches a component
the resource for the component gets used."""


def get_config():
  config = ConfigDict()

  config.learner = ConfigDict()
  config.learner.cpu = 1
  config.learner.mem = 0
  config.learner.gpu_compute = [1]
  config.learner.gpu_mem = [0]

  config.actor = ConfigDict()
  config.actor.cpu = 1
  config.actor.mem = 0
  config.actor.gpu_compute = []
  config.actor.gpu_mem = []

  config.replay = ConfigDict()
  config.replay.cpu = 1
  config.replay.mem = 0
  config.replay.gpu_compute = []
  config.replay.gpu_mem = []

  config.ps = ConfigDict(**config.replay)

  config.irs = ConfigDict()
  config.irs.cpu = 2
  config.irs.mem = 0
  config.irs.gpu_compute = []
  config.irs.gpu_mem = []

  config.tensorboard = ConfigDict(**config.irs)
  config.tensorboard.cpu = 2
  config.tensorboard.mem = 0

  config.tensorplex = ConfigDict(**config.irs)
  config.tensorplex.cpu = 1
  config.tensorplex.mem = 0

  return config