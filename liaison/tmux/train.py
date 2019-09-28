import argparse

from liaison.launch import hyper
from liaison.tmux.surreal_tmux import TurrealParser
from liaison.tmux.create_programs import build_program
from liaison.utils import ConfigDict
from liaison.scheduling import ScheduleManager
import argon

parser = argon.ArgumentParser('Resource Requirement config', add_help=False)
parser.add_argument('--n_actors', type=int, default=1)
parser.add_config_file(name='resource_req',
                       default='liaison/configs/resource_req.py')
parser.add_argument('--gpu_overload_obj_coeff', type=int, default=1)
parser.add_argument('--gpu_load_balancing_obj_coeff', type=int, default=1)
parser.add_argument('--gpu_wu_consolidation_obj_coeff', type=int, default=.25)
parser.add_argument('--cpu_overload_obj_coeff', type=int, default=1)
parser.add_argument('--cpu_load_balancing_obj_coeff', type=int, default=1)
parser.add_argument('--cpu_wu_consolidation_obj_coeff', type=int, default=10)


def train():
  tp = TurrealParser()
  tp.add_external_parser(parser)
  func, external_parser_args = tp.main()
  if func.__name__.split('action_')[-1] != 'create':
    return
  args = external_parser_args[0]

  cluster = tp.get_cluster()
  nodes = tp.get_nodes()

  # Specify experiment specific flags here.
  exp_flags = []
  exps = []
  for work_id, params in enumerate(
      hyper.discrete('agent_config.learning_rate', [1e-3])):
    exp = cluster.new_experiment('%s-%d' % (tp.experiment_name, work_id))
    build_program(exp, args.n_actors,
                  ConfigDict(argon.to_nested_dicts(args.resource_req_config)))

    exp_flag = ['--work_id', str(work_id)]
    exp_flag += ["--n_actors", str(args.n_actors)]
    exp_flag += hyper.to_commandline(params)
    exps.append(exp)
    exp_flags.append(exp_flag)

  exp_procs = [[
      proc for pg in exp.list_process_groups() for proc in pg.list_processes()
  ] + [proc for proc in exp.list_processes()] for exp in exps]
  print('-----------exp stats-------------')
  print('Number of work units: %d' % len(exps))
  print('Number of processes total: %d' % sum(map(len, exp_procs)))
  manager = ScheduleManager(
      nodes,
      exp_procs,
      gpu_overload_obj_coeff=args.gpu_overload_obj_coeff,
      gpu_load_balancing_obj_coeff=args.gpu_load_balancing_obj_coeff,
      gpu_wu_consolidation_obj_coeff=args.gpu_wu_consolidation_obj_coeff,
      cpu_overload_obj_coeff=args.cpu_overload_obj_coeff,
      cpu_load_balancing_obj_coeff=args.cpu_load_balancing_obj_coeff,
      cpu_wu_consolidation_obj_coeff=args.cpu_wu_consolidation_obj_coeff)

  ass, gpu_ass = manager.get_assignment()

  for wu_assignment, wu_gpu_assignment, procs in zip(ass, gpu_ass, exp_procs):
    for proc_id, (proc, server) in enumerate(zip(procs, wu_assignment)):
      proc.set_placement(nodes[server])
      if proc_id in wu_gpu_assignment:
        proc.set_gpus(wu_gpu_assignment[proc_id])

  tp.launch(exps, exp_flags)


def main():
  train()


if __name__ == '__main__':
  main()
