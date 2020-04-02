# Evaluate on a single MIP instance using the environment and the agent
# as primal heuristic in SCIP.
from pathlib import Path

import liaison.utils as U
from absl import app, logging
from argon import ArgumentParser, to_nested_dicts
from liaison.scip.evaluate import Evaluator
from liaison.utils import ConfigDict

parser = ArgumentParser()
# agent_config
parser.add_config_file(name='agent', required=True)

# env_config
parser.add_config_file(name='env', required=True)

# sess_config
parser.add_config_file(name='sess', required=True)

parser.add_argument('--n_local_moves', type=int, required=True)

parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('-n', '--name', required=True)
args = None


def main(argv):
  global args
  args = parser.parse_args(argv[1:])

  sess_config = ConfigDict(to_nested_dicts(args.sess_config))
  env_config = ConfigDict(to_nested_dicts(args.env_config))
  agent_config = ConfigDict(to_nested_dicts(args.agent_config))

  shell_class = U.import_obj(sess_config.shell.class_name,
                             sess_config.shell.class_path)
  env_class = U.import_obj(env_config.class_name, env_config.class_path)

  agent_class = U.import_obj(agent_config.class_name, agent_config.class_path)
  results_dir = Path(f'/data/nms/tfp/scip/{args.name}/')
  results_dir.mkdir(parents=True, exist_ok=True)

  evaluator = Evaluator(shell_class=shell_class,
                        shell_config=sess_config.shell,
                        agent_class=agent_class,
                        agent_config=agent_config,
                        env_class=env_class,
                        env_config=env_config,
                        seed=args.seed,
                        dataset=env_config.dataset,
                        dataset_type=env_config.dataset_type,
                        graph_start_idx=env_config.graph_start_idx,
                        batch_size=args.batch_size,
                        n_local_moves=args.n_local_moves,
                        results_dir=results_dir,
                        **sess_config)
  evaluator.run()


if __name__ == "__main__":
  app.run(main)
