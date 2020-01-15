import argparse
import math
import os
import pickle
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from absl import app
from liaison.daper.dataset_constants import DATASET_PATH, LENGTH_MAP
from liaison.daper.milp.dataset import MILP
from pyscipopt import (SCIP_HEURTIMING, SCIP_PARAMSETTING, SCIP_RESULT, Heur,
                       Model)

mpl.use('Agg')
plt.style.use('seaborn')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='milp-facilities-10')
parser.add_argument('--dataset_type', default='train')
parser.add_argument('--samples', nargs='+')
parser.add_argument('--timelimit', type=int)
args = None
N_SAMPLES = 8


class LogBestSol(Heur):

  def __init__(self, optimal_val):
    super(LogBestSol, self).__init__()
    self.optimal_val = optimal_val
    self.log = []

  def heurexec(self, heurtiming, nodeinfeasible):
    sol = self.model.getBestSol()
    obj = self.model.getSolObjVal(sol)
    self.log.append(float(obj) / self.optimal_val)
    return dict(result=SCIP_RESULT.DELAYED)


def scip_solve(mip, optimal_val):
  """Solves a mip/lp using scip"""
  solver = Model()
  solver.hideOutput()
  heur = LogBestSol(optimal_val)
  solver.includeHeur(heur,
                     "PyHeur",
                     "custom heuristic implemented in python",
                     "Y",
                     timingmask=SCIP_HEURTIMING.BEFORENODE)
  solver.setPresolve(SCIP_PARAMSETTING.OFF)
  mip.add_to_scip_solver(solver)
  if args.timelimit is not None:
    solver.setParam("limits/time", args.timelimit)

  solver.optimize()
  assert solver.getStatus() == 'optimal', solver.getStatus()
  obj = float(solver.getObjVal())
  return heur.log, obj


def testRelax(idx, milp):
  log_vals, obj = scip_solve(milp.mip, milp.optimal_objective)
  p = Path('/tmp') / f'server/results/{idx}.png'
  p.parent.mkdir(exist_ok=True, parents=True)
  fig, ax = plt.subplots()
  ax.plot(log_vals[1:], label=f'Best objective {obj}')
  ax.legend()
  fig.savefig(p)


def load_graph(dataset: str, dataset_type: str, graph_idx: int):
  path = DATASET_PATH[dataset]
  with open(os.path.join(path, dataset_type, '%d.pkl' % graph_idx), 'rb') as f:
    milp = pickle.load(f)
  return milp


def load_samples(dataset, dataset_type, samples):
  ret = []
  for i in map(int, samples):
    ret.append((i, load_graph(dataset, dataset_type, i)))
  return ret


def filter_samples(dataset: str, dataset_type: str):
  path = DATASET_PATH[dataset]
  l = []
  for fname in os.listdir(Path(path) / dataset_type):
    graph_idx = int(fname.rstrip('.pkl'))
    with open(os.path.join(path, dataset_type, f'{graph_idx}.pkl'), 'rb') as f:
      milp = pickle.load(f)
      l += [(milp.optimal_sol_metadata.n_nodes, graph_idx)]
  l = sorted(l, reverse=True)
  ret = []
  for pair in l[:N_SAMPLES]:
    _, i = pair
    print(i)
    ret.append((i, load_graph(dataset, dataset_type, i)))
  return ret


def main(argv):
  global args
  args = parser.parse_args(argv[1:])
  if args.samples:
    milps = load_samples(args.dataset, args.dataset_type, args.samples)
  else:
    milps = filter_samples(args.dataset, args.dataset_type)
  with Pool(N_SAMPLES) as pool:
    pool.starmap(testRelax, milps)


if __name__ == '__main__':
  app.run(main)
