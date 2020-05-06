import argparse
import os
import pickle
from math import fabs

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from liaison.daper.dataset_constants import (DATASET_INFO_PATH, DATASET_PATH,
                                             LENGTH_MAP,
                                             NORMALIZATION_CONSTANTS)
from liaison.daper.milp.primitives import ContinuousVariable
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', nargs='+', default=[], required=True)
args = parser.parse_args()

mpl.use('Agg')
plt.style.use('seaborn')


def read_pkl(fname):
  # run on liaison.csail.mit.edu node.
  fname = fname.replace('/data/', '/home/arc/vol/mnt/')
  with open(fname, 'rb') as f:
    return pickle.load(f)


def plt_cdf(ax, l, *args, **kwargs):
  x, y = sorted(l), np.arange(len(l)) / len(l)
  return ax.plot(x, y, *args, **kwargs)


def print_constants(dataset):
  stats = dict(
      constraint_rhs_normalizer=[],
      constraint_coeff_normalizer=[],
      obj_coeff_normalizer=[],
      obj_normalizer=[],
      optimal_var_vals=[],
      constraint_degree_normalizer=[],
      cont_variable_normalizer=[],
  )

  for i in trange(LENGTH_MAP[dataset]['train']):
    pkl = read_pkl(os.path.join(DATASET_PATH[dataset], f'train/{i}.pkl'))
    mip = pkl.mip
    stats['constraint_rhs_normalizer'].append(np.mean([c.rhs for c in mip.constraints]))
    stats['constraint_coeff_normalizer'].append(
        np.mean([np.mean(c.expr.coeffs) for c in mip.constraints]))
    stats['obj_coeff_normalizer'].append(np.mean(np.mean(mip.obj.expr.coeffs)))
    stats['obj_normalizer'].append(pkl.optimal_objective)
    stats['optimal_var_vals'].append(np.mean(list(pkl.optimal_solution.values())))
    stats['constraint_degree_normalizer'].append(np.max([len(c) for c in mip.constraints]))
    cont_vars = []
    for vname, v in mip.varname2var.items():
      if isinstance(v, ContinuousVariable):
        val = fabs(pkl.optimal_solution[vname])
        cont_vars.append(val)
    stats['cont_variable_normalizer'].append(np.mean(cont_vars))

  stats_max = dict(max_nodes=[], max_edges=[])
  # add valid and test as well for max nodes, edges
  for dataset_type in ['train', 'valid', 'test']:
    for i in trange(LENGTH_MAP[dataset][dataset_type]):
      pkl = read_pkl(os.path.join(DATASET_INFO_PATH[dataset], f'aux_info/{dataset_type}/{i}.pkl'))
      c_f, e_f, v_f = pkl['mip_features']
      stats_max['max_nodes'].append(len(v_f['var_names']) + len(c_f['values']) + 1)
      stats_max['max_edges'].append(len(e_f['values']))

  print(f"'\n{dataset}': dict(")
  for k, v in stats.items():
    print(f'{k}={fabs(np.mean(v))},')
  for k, v in stats_max.items():
    print(f'{k}={np.max(v)},')
  print('),')


if __name__ == '__main__':
  for dataset in args.datasets:
    print_constants(dataset)
    print('')  #new line
