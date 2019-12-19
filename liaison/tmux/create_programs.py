"""Script to create programs for distributed RL.
Does the following:

"""
import itertools


def get_fuzzy_match(config, name):
  for k, v in config.items():
    if name.startswith(k):
      return v
  raise Exception('No fuzzy match found for key %s' % name)


def build_program(exp,
                  n_actors,
                  res_req_config,
                  with_visualizers=True,
                  with_evaluators=True):
  learner = exp.new_process('learner')
  replay = exp.new_process('replay_worker-0')
  ps = exp.new_process('ps')
  irs = exp.new_process('irs')
  irs.set_hard_placement('os_csail')
  if with_visualizers:
    visualizers = exp.new_process('visualizers')
    visualizers.set_hard_placement('os_csail')
  else:
    visualizers = None
  actor_pg = exp.new_process_group('actor-*')
  actors = []
  for i in range(n_actors):
    actors.append(actor_pg.new_process('actor-{}'.format(i)))

  if with_evaluators:
    evaluator = exp.new_process('evaluators')
  else:
    evaluator = None

  setup_network(
      actors=actors,
      learner=learner,
      replay=replay,
      ps=ps,
      evaluator=evaluator,
      visualizers=visualizers,
      irs=irs,
  )
  for proc in [learner, replay, ps, irs, visualizers] + actors + [evaluator]:
    if proc:
      proc.set_costs(**get_fuzzy_match(res_req_config, proc.name))


def setup_network(*,
                  actors,
                  ps,
                  replay,
                  learner,
                  evaluator=None,
                  visualizers=None,
                  irs=None):
  """
        Sets up the communication between surreal
        components using symphony

        Args:
            actors, (list): list of symphony processes
            ps, replay, learner, visualizers:
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

  irs.binds('tensorplex')
  irs.binds('tensorplex-system')
  irs.binds('irs-frontend')
  irs.binds('irs-backend')

  for proc in itertools.chain(actors, [ps, replay, learner]):
    proc.connects('tensorplex')
    proc.connects('tensorplex-system')
    proc.connects('irs-frontend')

  if evaluator:
    evaluator.connects('tensorplex')
    evaluator.connects('irs-frontend')
    evaluator.connects('ps-frontend')

  if visualizers:
    visualizers.binds('visualizers-tb')
    visualizers.binds('visualizers-system-tb')
    visualizers.binds('visualizers-profiler-ui')
