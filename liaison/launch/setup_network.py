import itertools


def setup_network(*, agents, ps, replay, learner, tensorplex, loggerplex,
                  tensorboard, irs):
  """
        Sets up the communication between surreal
        components using symphony

        Args:
            agents, (list): list of symphony processes
            ps, replay, learner, tensorplex, loggerplex, tensorboard:
                symphony processes
    """
  for proc in agents:
    proc.connects('ps-frontend')
    proc.connects('collector-frontend')

  agents[0].binds('spec')

  ps.binds('ps-frontend')
  ps.binds('ps-backend')
  ps.connects('parameter-publish')

  replay.binds('collector-frontend')
  replay.binds('sampler-frontend')
  replay.binds('collector-backend')
  replay.binds('sampler-backend')

  learner.connects('spec')
  learner.connects('sampler-frontend')
  learner.binds('parameter-publish')
  learner.binds('prefetch-queue')

  tensorplex.binds('tensorplex')
  loggerplex.binds('loggerplex')

  irs.binds('irs')

  for proc in itertools.chain(agents, [ps, replay, learner]):
    proc.connects('tensorplex')
    proc.connects('loggerplex')
    proc.connects('irs')

  tensorboard.exposes({'tensorboard': 6006})
