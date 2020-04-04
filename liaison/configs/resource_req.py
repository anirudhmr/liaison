from liaison.utils import ConfigDict


"""If prefix of the process name matches a component
the resource for the component gets used."""


def get_config():
  config = ConfigDict()

  config.learner = ConfigDict()
  config.learner.cpu = 1
  config.learner.mem = 0
  config.learner.gpu_compute = []
  config.learner.gpu_mem = []
  config.learner.gpu_compute = [1]
  config.learner.gpu_mem = [11]

  config.actor = ConfigDict()
  config.actor.cpu = 1
  config.actor.mem = 0
  config.actor.gpu_compute = []
  config.actor.gpu_mem = []

  config.bundled_actor = ConfigDict()
  config.bundled_actor.cpu = 32
  config.bundled_actor.mem = 0
  config.bundled_actor.gpu_compute = [1]
  config.bundled_actor.gpu_mem = [11]

  config.evaluator = ConfigDict()
  config.evaluator.cpu = 1
  config.evaluator.mem = 0
  config.evaluator.gpu_compute = []
  config.evaluator.gpu_mem = []

  config.replay = ConfigDict()
  config.replay.cpu = 0
  config.replay.mem = 0  #50 * 1e3  # 50 GB
  config.replay.gpu_compute = []
  config.replay.gpu_mem = []

  config.ps = ConfigDict(**config.replay)

  config.irs = ConfigDict()
  config.irs.cpu = 2
  config.irs.mem = 0
  config.irs.gpu_compute = []
  config.irs.gpu_mem = []

  config.visualizers = ConfigDict(**config.irs)
  config.visualizers.cpu = 1
  config.visualizers.mem = 0

  config.tensorplex = ConfigDict(**config.irs)
  config.tensorplex.cpu = 1
  config.tensorplex.mem = 0

  return config
