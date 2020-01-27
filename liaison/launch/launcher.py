"""
Defines the LaunchSettings class that holds all the
information one needs to launch a component of surreal
"""
import copy
import faulthandler
import os
import subprocess
import sys
from argparse import ArgumentParser
from multiprocessing import Process
from threading import Thread

import liaison.utils as U
from argon import to_nested_dicts
from liaison.distributed import Learner, SimpleParameterServer
from liaison.irs import IRSClient, IRSServer
from liaison.loggers import (AvgPipeLogger, ConsoleLogger, DownSampleLogger,
                             KVStreamLogger, TensorplexLogger)
from liaison.replay import ReplayLoadBalancer
from liaison.utils import ConfigDict
from tensorplex import Loggerplex, Tensorplex

faulthandler.enable()


class Launcher:
  """
        Launchers are shared entrypoint for surreal experiments.
        Launchers define a main function that takes commandline
        arguments in the following way.
        `python launch_ppo.py <component_name> -- [additional_args]`
        component_name defines which part of the experiment should be
        run in this process
        [additional_args] should be shared among all involved processes
        to define behavior globally
    """

  def main(self):
    """
        The main function to be called
        ```
        if __name__ == '__main__':
            launcher = Launcher()
            launcher.main()
        ```
        """
    argv = sys.argv[1:]
    parser_args = argv
    config_args = []
    if '--' in argv:
      index = argv.index('--')
      parser_args = argv[:index]
      config_args = argv[index + 1:]
    parser = ArgumentParser(description='launch a surreal component')
    parser.add_argument('component_name',
                        type=str,
                        help='which component to launch')
    args, _ = parser.parse_known_args(parser_args)

    self.config_args = config_args

    self.setup(config_args)
    self.launch(args.component_name)

  def launch(self, component_name_in):
    if '-' in component_name_in:
      component_name, component_id = component_name_in.split('-')
      try:
        component_id = int(component_id)
      except ValueError:
        pass
    else:
      component_name = component_name_in
      component_id = None

    if component_name == 'actor':
      self.run_actor(actor_id=component_id)
    elif component_name == 'evaluator':
      self.run_evaluator(id=component_id)
    elif component_name == 'evaluators':
      self.run_evaluators()
    elif component_name == 'learner':
      self.run_learner()
    elif component_name == 'ps':
      self.run_ps()
    elif component_name == 'replay':
      self.run_replay()
    elif component_name == 'replay_loadbalancer':
      self.run_replay_loadbalancer()
    elif component_name == 'replay_worker':
      self.run_replay_worker(replay_id=component_id)
    elif component_name == 'visualizers':
      self.run_visualizers()
    elif component_name == 'irs':
      self.run_irs()
    else:
      raise ValueError('Unexpected component {}'.format(component_name))

  def run_component(self, component_name):
    # note that redirecting to sys.stdout instead of subprocess.PIPE
    # would mean that U.wait_for_popen would not be capturing the stderr
    # and some printing could get broken.
    return subprocess.Popen(
        [sys.executable, '-u', sys.argv[0], component_name, '--'] +
        self.config_args,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT)

  def _setup_actor_system_loggers(self, id):
    loggers = []
    # loggers.append(ConsoleLogger(name='system'))
    loggers.append(
        TensorplexLogger(
            client_id=f'actor/{id}',
            host=os.environ['SYMPH_TENSORPLEX_SYSTEM_HOST'],
            port=os.environ['SYMPH_TENSORPLEX_SYSTEM_PORT'],
            serializer=self.sess_config.tensorplex.serializer,
            deserializer=self.sess_config.tensorplex.deserializer,
        ))
    return loggers

  def run_actor(self, actor_id):
    """
        Launches an actor process with actor_id

    Args:
        actor_id (int): actor's id
    """

    agent_config, env_config, sess_config = (self.agent_config,
                                             self.env_config, self.sess_config)
    agent_class = U.import_obj(agent_config.class_name,
                               agent_config.class_path)

    shell_class = U.import_obj(sess_config.shell.class_name,
                               sess_config.shell.class_path)

    env_class = U.import_obj(env_config.class_name, env_config.class_path)

    shell_config = dict(agent_class=agent_class,
                        agent_config=agent_config,
                        **self.sess_config.shell)

    actor_config = dict(
        actor_id=actor_id,
        shell_class=shell_class,
        shell_config=shell_config,
        env_class=env_class,
        env_configs=[self.env_config] * self.batch_size,
        traj_length=self.traj_length,
        seed=self.seed + actor_id * self.batch_size,
        batch_size=self.batch_size,
        system_loggers=self._setup_actor_system_loggers(actor_id)
        if actor_id == 0 else [],
        **self.sess_config.actor)

    actor_class = U.import_obj(sess_config.actor.class_name,
                               sess_config.actor.class_path)
    actor_class(**actor_config)  # blocking constructor.

  def _setup_evaluator_loggers(self, evaluator_name):
    loggers = []
    loggers.append(
        AvgPipeLogger(ConsoleLogger(print_every=1, name=evaluator_name)))
    if evaluator_name in ['train', 'valid', 'test']:
      loggers.append(
          AvgPipeLogger(
              TensorplexLogger(
                  client_id=f'evaluator/%d' %
                  ['train', 'valid', 'test'].index(evaluator_name),
                  serializer=self.sess_config.tensorplex.serializer,
                  deserializer=self.sess_config.tensorplex.deserializer,
              )))
    loggers.append(
        KVStreamLogger(stream_id=evaluator_name, client=IRSClient(timeout=20)))
    return loggers

  def run_evaluator(self, id: str):
    env_config, sess_config, agent_config = (self.env_config, self.sess_config,
                                             self.agent_config)
    eval_config = self.eval_config

    agent_class = U.import_obj(agent_config.class_name,
                               agent_config.class_path)
    shell_class = U.import_obj(sess_config.shell.class_name,
                               sess_config.shell.class_path)
    env_class = U.import_obj(env_config.class_name, env_config.class_path)
    agent_config = copy.deepcopy(agent_config)
    agent_config.update(evaluation_mode=True)
    shell_config = dict(agent_class=agent_class,
                        agent_config=agent_config,
                        **self.sess_config.shell)

    env_configs = []

    print('**********************************************')
    print('WARNING: graph_start_idx is switched off')
    print('**********************************************')
    for i in range(eval_config.batch_size):
      env_config = ConfigDict(**self.env_config)
      env_config.update({
          eval_config.dataset_type_field: id,
          'graph_start_idx': 0,
          **eval_config.env_config
      })
      env_configs.append(env_config)

    evaluator_config = dict(
        shell_class=shell_class,
        shell_config=shell_config,
        env_class=env_class,
        env_configs=env_configs,
        traj_length=self.traj_length,
        loggers=self._setup_evaluator_loggers(id),
        heuristic_loggers=self._setup_evaluator_loggers(f'heuristic-{id}'),
        seed=self.seed,
        **eval_config)
    from liaison.distributed import Evaluator
    Evaluator(**evaluator_config)

  def run_evaluators(self):
    components = [
        self.run_component(f'evaluator-{eval_type}')
        for eval_type in ['train', 'valid', 'test']
    ]
    U.wait_for_popen(components)

  def _setup_learner_loggers(self):
    loggers = []
    loggers.append(ConsoleLogger(print_every=5))
    loggers.append(
        TensorplexLogger(
            client_id='learner/learner',
            serializer=self.sess_config.tensorplex.serializer,
            deserializer=self.sess_config.tensorplex.deserializer,
        ))
    return loggers

  def _setup_learner_system_loggers(self):
    loggers = []
    # loggers.append(ConsoleLogger(name='system'))
    loggers.append(
        TensorplexLogger(
            client_id='learner/learner',
            host=os.environ['SYMPH_TENSORPLEX_SYSTEM_HOST'],
            port=os.environ['SYMPH_TENSORPLEX_SYSTEM_PORT'],
            serializer=self.sess_config.tensorplex.serializer,
            deserializer=self.sess_config.tensorplex.deserializer,
        ))
    return loggers

  def run_learner(self, iterations=None):
    """
        Launches the learner process.
        Learner consumes experience from replay
        and publishes experience to parameter server
    """

    agent_class = U.import_obj(self.agent_config.class_name,
                               self.agent_config.class_path)
    learner = Learner(agent_class=agent_class,
                      agent_config=self.agent_config,
                      traj_length=self.traj_length,
                      seed=self.seed,
                      loggers=self._setup_learner_loggers(),
                      system_loggers=self._setup_learner_system_loggers(),
                      **self.sess_config.learner)
    learner.main()

  def run_ps(self):
    """
        Lauches the parameter server process.
        Serves parameters to agents
    """
    server = SimpleParameterServer(
        publish_port=os.environ['SYMPH_PS_PUBLISHING_PORT'],
        serving_port=os.environ['SYMPH_PS_SERVING_PORT'])
    server.start()
    server.join()

  def run_replay(self):
    """
        Launches the replay process.
        Replay collects experience from agents
        and serve them to learner
    """
    loadbalancer = self.run_component('replay_loadbalancer')
    components = [loadbalancer]
    for replay_id in range(self.sess_config.replay.n_shards):
      component_name = 'replay_worker-{}'.format(replay_id)
      replay = self.run_component(component_name)
      components.append(replay)
    U.wait_for_popen(components)

  def run_replay_loadbalancer(self):
    """
            Launches the learner and agent facing load balancing proxys
            for replays
        """
    loadbalancer = ReplayLoadBalancer()
    loadbalancer.launch()
    loadbalancer.join()

  def run_replay_worker(self, replay_id):
    """
            Launches a single replay server

        Args:
            replay_id: The id of the replay server
        """

    replay_class = U.import_obj(self.sess_config.replay.class_name,
                                self.sess_config.replay.class_path)

    replay = replay_class(seed=self.seed,
                          index=replay_id,
                          tensorplex_config=self.sess_config.tensorplex,
                          **self.sess_config.replay)
    replay.start_threads()
    replay.join()

  def run_visualizers(self):
    """
        Launches the following visualization processes:
          tensorboard
          systemboard
          profiler_ui
    """
    procs = []
    # Visualize all work units with tensorboard.
    folder = os.path.join(self.results_folder, 'tensorplex_metrics')
    cmd = [
        'tensorboard', '--logdir', folder, '--port',
        str(os.environ['SYMPH_VISUALIZERS_TB_PORT'])
    ]
    procs += [subprocess.Popen(cmd)]

    # launch systemboard
    folder = os.path.join(self.results_folder, 'tensorplex_system_profiles')
    cmd = [
        'tensorboard', '--logdir', folder, '--port',
        str(os.environ['SYMPH_VISUALIZERS_SYSTEM_TB_PORT'])
    ]
    procs += [subprocess.Popen(cmd)]

    # launch profile viewer
    cmd = [
        'python', '-m', 'profiler_ui.ui', '--profile_context_path',
        os.path.join(self.results_folder, 'tensorflow_profiles',
                     str(self.work_id), 'timeline.json'), '--port',
        str(os.environ['SYMPH_VISUALIZERS_PROFILER_UI_PORT'])
    ]
    procs += [subprocess.Popen(cmd)]
    # block for the above processes
    U.wait_for_popen(procs)

  def _start_tensorplex(self):
    """
        Launches a tensorplex process.
        It receives data from multiple sources and
        send them to tensorboard.
    """
    folder1 = os.path.join(self.results_folder, 'tensorplex_metrics',
                           str(self.work_id))
    folder2 = os.path.join(self.results_folder, 'tensorplex_system_profiles',
                           str(self.work_id))
    tensorplex_config = self.sess_config.tensorplex
    threads = []

    for folder, port in zip([folder1, folder2], [
        os.environ['SYMPH_TENSORPLEX_PORT'],
        os.environ['SYMPH_TENSORPLEX_SYSTEM_PORT']
    ]):
      tensorplex = Tensorplex(
          folder,
          max_processes=tensorplex_config.max_processes,
      )
      """
        Tensorboard categories:
          learner/replay/eval: algorithmic level, e.g. reward, ...
          ***-core: low level metrics, i/o speed, computation time, etc.
          ***-system: Metrics derived from raw metric data in core,
                      i.e. exp_in/exp_out
      """
      tensorplex.register_normal_group('learner').register_indexed_group(
          'actor', tensorplex_config.agent_bin_size).register_indexed_group(
              'replay',
              100).register_indexed_group('ps', 100).register_indexed_group(
                  'evaluator', tensorplex_config.agent_bin_size)

      thread = Thread(target=tensorplex.start_server,
                      kwargs=dict(port=port,
                                  serializer=tensorplex_config.serializer,
                                  deserializer=tensorplex_config.deserializer))
      thread.start()
      threads.append(thread)

    return threads

  def run_irs(self):
    tensorplex_threads = self._start_tensorplex()
    self._irs_server = IRSServer(
        results_folder=self.results_folder,
        agent_config=self.agent_config,
        env_config=self.env_config,
        sess_config=self.sess_config,
        exp_name=self.experiment_name,
        exp_id=self.experiment_id,
        work_id=self.work_id,
        configs_folder=os.path.join(self.results_folder, 'config',
                                    str(self.work_id)),
        src_folder=os.path.join(self.results_folder, 'src', str(self.work_id)),
        checkpoint_folder=os.path.join(self.results_folder, 'checkpoints',
                                       str(self.work_id)),
        profile_folder=os.path.join(self.results_folder, 'tensorflow_profiles',
                                    str(self.work_id)),
        cmd_folder=os.path.join(self.results_folder, 'cmds',
                                str(self.work_id)),
        kvstream_folder=os.path.join(self.results_folder, 'kvstream',
                                     str(self.work_id)),
        hyper_param_config_file=os.path.join(self.results_folder,
                                             'hyper_params', str(self.work_id),
                                             'hyper_params.json'),
        hyper_params=self.hyper_params,
        **self.sess_config.irs)
    self._irs_server.launch()
    self._irs_server.join()
    for thread in tensorplex_threads:
      thread.join()
