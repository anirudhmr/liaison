import argparse
import os
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', required=True)
parser.add_argument('-o', '--output_dir', required=True)
parser.add_argument('-e', '--exp_ids', nargs='+')
args = parser.parse_args()


def get_exp_name(exp_id):
  return os.listdir(f'{args.input_dir}/{exp_id}/')[0]


def f(exp_id):
  d = f'{args.input_dir}/{exp_id}/{get_exp_name(exp_id)}/kvstream/'
  dst = f'{args.output_dir}/{exp_id}/{get_exp_name(exp_id)}/kvstream/'
  for wu in os.listdir(d):
    fnames = list(filter(lambda k: k.endswith('.pkl'), os.listdir(f'{d}/{wu}')))
    eval_types = {}  # train, valid, test, heuristic-train etc.
    for fname in fnames:
      fname = fname.split('.pkl')[0]
      eval_type = '-'.join(fname.split('-')[:-1])
      l = eval_types.get(eval_type, [])
      l.append(int(fname.split('-')[-1]))
      eval_types[eval_type] = list(sorted(l))

    for eval_type, l in eval_types.items():
      Path(f'{dst}/{wu}').mkdir(exist_ok=True, parents=True)
      shutil.copy2(f'{d}/{wu}/{eval_type}-{l[-1]}.pkl', f'{dst}/{wu}/{eval_type}-{l[-1]}.pkl')

  # copy hyper_params
  shutil.copytree(f'{args.input_dir}/{exp_id}/{get_exp_name(exp_id)}/hyper_params/',
                  f'{args.output_dir}/{exp_id}/{get_exp_name(exp_id)}/hyper_params')


if __name__ == '__main__':
  for exp_id in args.exp_ids:
    f(exp_id)
