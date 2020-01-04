import numpy as np
from liaison.daper import ConfigDict
from liaison.env import StepType
from liaison.env.rins import Env


def run(n_trials, seeds, env):
  assert len(seeds) == n_trials

  log_vals = [[] for _ in range(n_trials)]
  for trial_i, seed in zip(range(n_trials), seeds):
    rng = np.random.RandomState(seed)
    ts = env.reset()
    obs = ConfigDict(ts.observation)
    log_vals[trial_i].append(obs.curr_episode_log_values)

    while ts.step_type != StepType.LAST:
      act = rng.choice(len(obs.mask), 1, p=obs.mask / np.sum(obs.mask))
      ts = env.step(act)
      obs = ConfigDict(ts.observation)
      if obs.graph_features.globals[Env.GLOBAL_LOCAL_SEARCH_STEP]:
        log_vals[trial_i].append(obs.curr_episode_log_values)
  return log_vals
