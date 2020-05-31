import argparse
import pickle
import traceback
from math import fabs
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from liaison.daper.dataset_constants import DATASET_PATH
from liaison.scripts.plts.mpl_config import *

mpl.use('Agg')
mpl.style.use('seaborn')
mpl.rcParams.update(mpl_params)
sns = set_seaborn_styles(sns)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True)
parser.add_argument('-n', '--name', required=True)
parser.add_argument('--dtype', default='test')
parser.add_argument('--results_dir', '-r', default='/data/nms/tfp/evaluation/')
parser.add_argument('-o', '--output_dir', default='/data/nms/tfp/paper/figs/')
parser.add_argument('-N', '--n_samples', default=32)
parser.add_argument('--aspect_ratio', '-a', type=float, default=None)
parser.add_argument('--line-width', '-l', type=float, default=mpl_linewidth)
parser.add_argument('--paper',
                    '-p',
                    action='store_true',
                    help='Generate for paper mode as opposed to debug mode.')
parser.add_argument('--no_legend', action='store_true')
parser.add_argument('--ignore_exceptions', '-i', action='store_true')
args = parser.parse_args()


def load_pickle(p):
  with open(f'{args.results_dir}/{p}', 'rb') as f:
    return pickle.load(f)


def load_sample(graph):
  with open(f'{DATASET_PATH[args.dataset]}/{args.dtype}/{graph}.pkl', 'rb') as f:
    return pickle.load(f)


def plt_cdf(ax, l, *cmd_args, label=None, **kwargs):
  x, y = sorted(l), np.arange(len(l)) / len(l)
  if label and not args.paper:
    label += ' (Avg: %.2f)' % np.mean(x)
  if not args.no_legend:
    kwargs['label'] = label
  return ax.plot(x, y, *cmd_args, **kwargs)


def set_aspect_ratio(ax, aspect_ratio):
  xleft, xright = ax.get_xlim()
  ybottom, ytop = ax.get_ylim()
  # the abs method is used to make sure that all numbers are positive
  # because x and y axis of an axes maybe inversed.
  ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * aspect_ratio)


def primal_gap(curr_obj, optimal_obj):
  if curr_obj == 0 and optimal_obj == 0:
    rew = 0.
  elif np.sign(curr_obj) * np.sign(optimal_obj) < 0:
    rew = 1.
  else:
    rew = fabs(optimal_obj - curr_obj) / max(fabs(optimal_obj), fabs(curr_obj))
  return rew


def compute_primal_integral(data):
  return sum(data['quality'])


def plt_empty_fig(xlabel):
  plt.figure()
  plt.xlabel(xlabel)
  plt.ylabel(f'CDF Probability')
  if args.aspect_ratio is not None:
    set_aspect_ratio(plt.gca(), args.aspect_ratio)
  plt.tight_layout()
  save_path = Path(args.output_dir) / Path(f'{args.name}/standalone/empty-{xlabel}.png')
  save_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(save_path, bbox_inches='tight')


def plt_fn():
  agent_pis = []
  li_pis = []
  mi_pis = []
  rins_pis = []
  random_pis = []

  agent_final_gaps = []
  li_final_gaps = []
  mi_final_gaps = []
  rins_final_gaps = []
  random_final_gaps = []

  for graph in range(args.n_samples):
    try:
      agent = load_pickle(f'standalone/agent/{args.name}/{graph}/out.pkl')
      rins = load_pickle(f'standalone/rins/{args.name}/{graph}/out.pkl')
      random = load_pickle(f'standalone/random/{args.name}/{graph}/out.pkl')
    except Exception as e:
      if args.ignore_exceptions:
        continue
      raise e

    sample = load_sample(graph)

    agent_pis.append(compute_primal_integral(agent))
    # li_pis.append(compute_primal_integral(l2))
    # mi_pis.append(compute_primal_integral(l3))
    random_pis.append(compute_primal_integral(random))
    rins_pis.append(compute_primal_integral(rins))

    agent_final_gaps.append(agent['quality'][-1])
    # li_final_gaps.append(l2[-2][-1])
    # mi_final_gaps.append(l3[-2][-1])
    random_final_gaps.append(random['quality'][-1])
    rins_final_gaps.append(rins['quality'][-1])

  kwargs = {'linewidth': args.line_width}
  plt.figure()
  plt_cdf(plt.gca(), agent_pis, label=f'Agent', **kwargs)
  # plt_cdf(plt.gca(), li_pis, label=f'Least Integral: {np.mean(li_pis)}')
  # plt_cdf(plt.gca(), mi_pis, label=f'Most Integral: {np.mean(mi_pis)}')
  plt_cdf(plt.gca(), random_pis, label=f'Random', **kwargs)
  plt_cdf(plt.gca(), rins_pis, label=f'RINS', **kwargs)
  if args.paper:
    plt.xlabel('Primal Integral')
  else:
    plt.xlabel(f'Primal Integral - # of samples: {len(agent_pis)}')
  plt.ylabel(f'CDF Probability')
  if args.aspect_ratio is not None:
    set_aspect_ratio(plt.gca(), args.aspect_ratio)

  plt.legend()
  plt.tight_layout()
  save_path = Path(args.output_dir) / Path(f'{args.name}/standalone/pi.png')
  save_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(save_path, bbox_inches='tight')

  plt.figure()
  plt_cdf(plt.gca(), agent_final_gaps, label=f'Agent', **kwargs)
  # plt_cdf(plt.gca(), li_final_gaps, label=f'Least Integral: {np.mean(li_final_gaps)}')
  # plt_cdf(plt.gca(), mi_final_gaps, label=f'Most Integral: {np.mean(mi_final_gaps)}')
  plt_cdf(plt.gca(), random_final_gaps, label=f'Random', **kwargs)
  plt_cdf(plt.gca(), rins_final_gaps, label=f'RINS', **kwargs)
  if args.paper:
    plt.xlabel('Final Primal Gap')
  else:
    plt.xlabel(f'Final Primal Gap - # of samples: {len(agent_final_gaps)}')
  plt.ylabel(f'CDF Probability')
  plt.legend()
  plt.tight_layout()
  if args.aspect_ratio is not None:
    set_aspect_ratio(plt.gca(), args.aspect_ratio)
  save_path = Path(args.output_dir) / Path(f'{args.name}/standalone/pg.png')
  save_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
  plt_fn()
  plt_empty_fig('Primal Integral')
  plt_empty_fig('Final Primal Gap')
