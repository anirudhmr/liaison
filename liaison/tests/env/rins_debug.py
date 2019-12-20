# debug assert fails during training in rins
# by trying to find the cases where this happens using a uniform random
# agent against the rins environment.
import argparse
import os
import pdb
import pickle
import signal
import sys

import liaison.utils as U
import numpy as np
import tensorflow as tf
import tree as nest
from absl import app
from liaison.agents.ur_discrete import Agent
from liaison.env import StepType
from liaison.env.batch import ParallelBatchedEnv, SerialBatchedEnv
from liaison.env.rins import Env
from liaison.utils import ConfigDict

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--use_procs', action='store_true')
parser.add_argument('--k', type=int, default=20)
parser.add_argument('--run_debug', action='store_true')
parser.add_argument('--debug_actions_fname')
parser.add_argument('--debug_actions_index', type=int)
parser.add_argument('--graph_start_idx', type=int, default=4070)
global args


def get_env_config():
  """get rins env config."""
  config = ConfigDict()

  # required fields.
  config.class_path = "liaison.env.rins"  # should be rel to the parent directory.
  config.class_name = "Env"

  # makes observations suitable for the MLP model.
  config.make_obs_for_mlp = True
  # adds all the constraints to MLP state space.
  # adds #variables * #constraints dimensions to the state space.
  config.mlp_embed_constraints = False

  config.make_obs_for_self_attention = False
  """if graph_seed < 0, then use the environment seed"""
  config.graph_seed = 42

  config.dataset = 'milp-facilities-10'
  config.dataset_type = 'train'
  config.graph_start_idx = args.graph_start_idx
  config.n_graphs = 1

  config.k = args.k
  config.steps_per_episode = 2000

  return config


def make_env():
  env_config = get_env_config()
  return ParallelBatchedEnv(args.bs,
                            U.import_obj(env_config.class_name,
                                         env_config.class_path),
                            [env_config] * args.bs,
                            args.seed,
                            use_threads=not args.use_procs)


def make_agent(action_spec):
  agent = Agent('test', action_spec, args.seed)
  return agent


def make_ph(ts):

  def mk_ph(spec):
    return tf.placeholder(dtype=spec.dtype, shape=spec.shape)

  ts_ph = nest.map_structure(mk_ph, ts.step_type)
  obs_ph = nest.map_structure(mk_ph, ts.observation)
  return ts_ph, obs_ph


def get_feed_dict(ts, ts_ph, obs_ph):
  d = {
      p: v
      for p, v in zip(nest.flatten(obs_ph), nest.flatten(ts.observation))
  }
  d.update({ts_ph: ts.step_type})
  return d


def log_debug_actions(actions):
  with open(f'/tmp/{os.getpid()}.pkl', 'wb') as f:
    pickle.dump(dict(actions=actions), f)


class timeout:

  def __init__(self, seconds=1, error_message='Timeout'):
    self.seconds = seconds
    self.error_message = error_message

  def handle_timeout(self, signum, frame):
    raise TimeoutError(self.error_message)

  def __enter__(self):
    signal.signal(signal.SIGALRM, self.handle_timeout)
    signal.alarm(self.seconds)

  def __exit__(self, type, value, traceback):
    signal.alarm(0)


def read_pkl(fname):
  with open(fname, 'rb') as f:
    return pickle.load(f)


def run_debug():
  acts = read_pkl(args.debug_actions_fname)['actions']
  idx = args.debug_actions_index

  env_config = get_env_config()
  env = U.import_obj(env_config.class_name,
                     env_config.class_path)(**env_config,
                                            seed=args.seed,
                                            id=idx)

  ts = env.reset()
  for act in acts[idx]:
    env.step(act)


def run_diag():
  env = make_env()
  print('Env created!')

  ts = env.reset()
  print('Env reset!')

  agent = make_agent(env.action_spec())
  ts_ph, obs_ph = make_ph(ts)
  step_out_op = agent.step(step_type=ts_ph,
                           reward=None,
                           obs=obs_ph,
                           prev_state=None)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  i = 0
  actions = [[] for _ in range(args.bs)]
  print(f'Logs PID: {os.getpid()}')
  while True:
    if i % int(1e3) == 0:
      print('.', end='')
      sys.stdout.flush()

    for i, step_type in enumerate(ts.step_type):
      if step_type == StepType.FIRST:
        actions[i] = []

    step_output = sess.run(step_out_op,
                           feed_dict=get_feed_dict(ts, ts_ph, obs_ph))
    for i, act in enumerate(step_output.action):
      actions[i].append(act)

    try:
      with timeout(10):
        ts = env.step(step_output.action)
    except TimeoutError:
      print('Error detected...')
      log_debug_actions(actions)

    i += 1


def main(argv):
  global args
  args = parser.parse_args(argv[1:])
  if args.run_debug:
    run_debug()
  else:
    run_diag()


if __name__ == '__main__':
  app.run(main)
