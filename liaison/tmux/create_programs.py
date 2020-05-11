"""Script to create programs for distributed RL.
Does the following:

"""
import itertools

import liaison.utils as U


def get_fuzzy_match(config, name):
  for k, v in config.items():
    if name.startswith(k):
      return v
  raise Exception('No fuzzy match found for key %s' % name)


def build_program(
    exp,
    n_actors,
    res_req_config,
    bundle_actors,
    irs_placement,
    visualizer_placement,
    with_visualizers=True,
    with_evaluators=True,
    without_valid_and_test_evaluators=False,
    with_irs_proxy=False,
    irs_proxy_placement=None,
):
  learner = exp.new_process('learner')
  replay = exp.new_process('replay_worker-0')
  ps = exp.new_process('ps')
  irs = exp.new_process('irs')
  irs.set_hard_placement(irs_placement)

  if with_irs_proxy:
    irs_proxy = exp.new_process('irs_proxy')
    irs_proxy.set_hard_placement(irs_proxy_placement)
  else:
    irs_proxy = None

  if with_visualizers:
    visualizers = exp.new_process('visualizers')
    visualizers.set_hard_placement(visualizer_placement)
  else:
    visualizers = None

  actors = []
  if bundle_actors:
    actors.append(exp.new_process('bundled_actor'))
  else:
    actor_pg = exp.new_process_group('actor-*')
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
      irs_proxy=irs_proxy,
  )
  for proc in [learner, replay, ps, irs, irs_proxy, visualizers] + actors + [evaluator]:
    if proc:
      proc.set_costs(**get_fuzzy_match(res_req_config, proc.name))

  # define coloc_constraints
  coloc_constraints = ['learner;ps;replay_worker-0']
  return coloc_constraints


def setup_network(*,
                  actors,
                  ps,
                  replay,
                  learner,
                  evaluator=None,
                  visualizers=None,
                  irs=None,
                  irs_proxy=None):
  """
    Sets up the communication between surreal
    components using symphony

    Args:
        actors, (list): list of symphony processes
        ps, replay, learner, visualizers:
            symphony processes
  """
  for proc in actors:
    proc.connects('ps-serving')
    proc.connects('collector-frontend')

  actors[0].binds('spec')

  ps.binds('ps-publishing')
  ps.binds('ps-serving')

  replay.binds('collector-frontend')
  replay.binds('sampler-frontend')
  replay.binds('collector-backend')
  replay.binds('sampler-backend')

  learner.connects('spec')
  learner.connects('sampler-frontend')
  learner.binds('ps-publishing')
  learner.binds('prefetch-queue')

  irs.binds('tensorplex')
  irs.binds('tensorplex-var')
  irs.binds('tensorplex-system')
  irs.binds('irs')

  if irs_proxy:
    irs_proxy.binds('irs-proxy')
    irs_proxy.connects('irs')

  for proc in itertools.chain(actors, [ps, replay, learner]):
    proc.connects('tensorplex')
    proc.connects('tensorplex-system')
    proc.connects('tensorplex-var')
    proc.connects('irs')
    if irs_proxy:
      proc.connects('irs-proxy')

  if evaluator:
    evaluator.connects('tensorplex')
    evaluator.connects('irs')
    if irs_proxy:
      evaluator.connects('irs-proxy')
    evaluator.connects('ps-serving')

  if visualizers:
    visualizers.binds('visualizers-tb')
    visualizers.binds('visualizers-system-tb')
    visualizers.binds('visualizers-profiler-ui')
    visualizers.binds('visualizers-var-tb')
