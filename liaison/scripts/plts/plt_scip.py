import argparse
import functools
import pickle
import traceback
from math import fabs
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from absl import app
from liaison.daper.dataset_constants import DATASET_PATH
from liaison.scripts.plts.mpl_config import *
from tqdm import trange

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
parser.add_argument('--extension', default='png')
parser.add_argument('--scip', action='store_true')
global args


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


def extract(data, sample):
  l = []
  for _, obj, step, *_ in data:
    if obj is not None:
      l.append((step, primal_gap(obj, sample['optimal_objective'])))
  assert obj is None
  l.append((step, None))
  return l


def compute_primal_integral(l):
  if args.scip:
    integral = 0.
    for idx, (i, gap) in enumerate(l[:-1]):
      integral += (gap * (l[idx + 1][0] - i))
    return integral
  return sum(l['quality'])


def save_fig(fname):
  save_path = Path(fname)
  save_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(fname + '.png', bbox_inches='tight')


def plt_fn():
  agent_pis = []
  no_agent_pis = []
  li_pis = []
  mi_pis = []
  rins_pis = []
  random_pis = []

  agent_final_gaps = []
  no_agent_final_gaps = []
  li_final_gaps = []
  mi_final_gaps = []
  rins_final_gaps = []
  random_final_gaps = []

  for graph in trange(args.n_samples):
    try:
      if args.scip:
        agent = load_pickle(f'scip/{args.name}/{graph}/out.pkl')
        rins = load_pickle(f'rins/{args.name}/{graph}/out.pkl')
        random = load_pickle(f'random/{args.name}/{graph}/out.pkl')
        no_agent = load_pickle(f'without_agent/{args.name}/{graph}/out.pkl')
      else:
        agent = load_pickle(f'standalone/agent/{args.name}/{graph}/out.pkl')
        rins = load_pickle(f'standalone/rins/{args.name}/{graph}/out.pkl')
        random = load_pickle(f'standalone/random/{args.name}/{graph}/out.pkl')
    except Exception as e:
      if args.ignore_exceptions:
        continue
      raise e

    if args.scip:
      f = functools.partial(extract, sample=load_sample(graph))
      agent = f(agent)
      # Ignore cases where no feasible solution found
      if len(agent) <= 1: continue
      rins = f(rins)
      random = f(random)
      no_agent = f(no_agent)

    agent_pis.append(compute_primal_integral(agent))
    random_pis.append(compute_primal_integral(random))
    rins_pis.append(compute_primal_integral(rins))
    if args.scip: no_agent_pis.append(compute_primal_integral(no_agent))

    if args.scip:
      agent_final_gaps.append(agent[-2][-1])
      random_final_gaps.append(random[-2][-1])
      rins_final_gaps.append(rins[-2][-1])
      no_agent_final_gaps.append(no_agent[-2][-1])
    else:
      agent_final_gaps.append(agent['quality'][-1])
      random_final_gaps.append(random['quality'][-1])
      rins_final_gaps.append(rins['quality'][-1])

  def plt_cdfs(key, ys, labels, ylabel):
    kwargs = {'linewidth': args.line_width}
    plt.figure()
    for y, label, color in zip(ys, labels, COLORS):
      plt_cdf(plt.gca(), y, label=label, color=color, **kwargs)
    if args.paper:
      plt.xlabel(ylabel)
    else:
      plt.xlabel(f'{ylabel} - # of samples: {len(agent_pis)}')
    plt.ylabel(f'CDF Probability')
    if args.aspect_ratio is not None:
      set_aspect_ratio(plt.gca(), args.aspect_ratio)
    if not args.no_legend:
      plt.legend()
    plt.tight_layout()
    save_fig(f'{args.output_dir}/{args.name}/%s/{key}' % ('scip' if args.scip else 'standalone'))

  def plt_bars(key, ys, labels, ylabel):
    kwargs = {
        'axes.labelsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
    }
    mpl.rcParams.update(kwargs)
    plt.figure(figsize=[2.5, 5])
    width = 0.1  # the width of the bars
    xs = width / 2. + width * np.arange(len(labels))  # the label locations

    for i, (label, x, y, color) in enumerate(zip(labels, xs, ys, COLORS)):
      plt.bar(x, np.mean(y), width, label=label, color=color)
      plt.errorbar(x, np.mean(y), yerr=[[0], [np.std(y)]], capsize=5, capthick=1, color=color)

    if not args.no_legend:
      plt.legend()
    padding = 1.5 * width
    plt.xlim(-padding, xs[-1] + padding)
    # plt.gca().axes.get_xaxis().set_visible(False)
    plt.xticks([], [])
    plt.xlabel('$\it{cauction}$')
    plt.ylabel(ylabel)
    save_fig(f'{args.output_dir}/{args.name}/%s/hist_{key}' %
             ('scip' if args.scip else 'standalone'))

  labels = ['Agent', 'Random', 'RINS']
  if args.scip:
    labels += ['No heuristic']
  for f in (plt_bars, plt_cdfs):
    mpl.rcParams.update(mpl_params)
    if args.scip:
      f('pg', [agent_final_gaps, random_final_gaps, rins_final_gaps, no_agent_final_gaps], labels,
        'Primal Gap')
      f('pi', [agent_pis, random_pis, rins_pis, no_agent_pis], labels, 'Primal Integral')
    else:
      f('pg', [agent_final_gaps, random_final_gaps, rins_final_gaps], labels, 'Primal Gap')
      f('pi', [agent_pis, random_pis, rins_pis], labels, 'Primal Integral')


def main(argv):
  global args
  args = parser.parse_args(argv[1:])
  plt_fn()


if __name__ == '__main__':
  app.run(main)
