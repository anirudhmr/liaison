import itertools


def setup_network(*,
                  actors,
                  ps,
                  replay,
                  learner,
                  tensorplex,
                  loggerplex,
                  tensorboard=None,
                  irs=None):
  """
        Sets up the communication between surreal
        components using symphony

        Args:
            actors, (list): list of symphony processes
            ps, replay, learner, tensorplex, loggerplex, tensorboard:
                symphony processes
    """
  for proc in actors:
    proc.connects('ps-frontend')
    proc.connects('collector-frontend')

  actors[0].binds('spec')

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

  # irs.binds('irs-frontend')
  # irs.binds('irs-backend')

  for proc in itertools.chain(actors, [ps, replay, learner]):
    proc.connects('tensorplex')
    proc.connects('loggerplex')
    # proc.connects('irs-frontend')

  if tensorboard:
    tensorboard.exposes({'tensorboard': 6006})
