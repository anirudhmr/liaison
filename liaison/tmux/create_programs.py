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


def build_program(exp,
                  n_actors,
                  res_req_config,
                  bundle_actors,
                  with_visualizers=True,
                  with_evaluators=True):
  learner = exp.new_process('learner')
  replay = exp.new_process('replay_worker-0')
  ps = exp.new_process('ps')
  irs = exp.new_process('irs')
  irs.set_hard_placement('cloudlab_clemson_clnode2_0')
  if with_visualizers:
    visualizers = exp.new_process('visualizers')
    visualizers.set_hard_placement('cloudlab_clemson_clnode2_0')
  else:
    visualizers = None

  actors = []
  if bundle_actors:
    actors.append(exp.new_process('actor_bundle'))
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
  irs.binds('tensorplex-system')
  irs.binds('irs')

  for proc in itertools.chain(actors, [ps, replay, learner]):
    proc.connects('tensorplex')
    proc.connects('tensorplex-system')
    proc.connects('irs')

  if evaluator:
    evaluator.connects('tensorplex')
    evaluator.connects('irs')
    evaluator.connects('ps-serving')

  if visualizers:
    visualizers.binds('visualizers-tb')
    visualizers.binds('visualizers-system-tb')
    visualizers.binds('visualizers-profiler-ui')
