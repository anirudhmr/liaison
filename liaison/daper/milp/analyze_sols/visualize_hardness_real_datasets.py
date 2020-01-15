import argparse
import math
import os
import pickle
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from absl import app
from liaison.daper.dataset_constants import DATASET_PATH, LENGTH_MAP
from liaison.daper.milp.dataset import MILP
from pyscipopt import (SCIP_HEURTIMING, SCIP_PARAMSETTING, SCIP_RESULT, Heur,
                       Model)

mpl.use('Agg')
plt.style.use('seaborn')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--n_samples', type=int)
parser.add_argument('--timelimit', type=float)
parser.add_argument('--gap', type=float, default=0.)
args = None


class LogBestSol(Heur):

  def __init__(self):
    super(LogBestSol, self).__init__()
    self.log = []

  def heurexec(self, heurtiming, nodeinfeasible):
    sol = self.model.getBestSol()
    obj = self.model.getSolObjVal(sol)
    self.log.append(float(obj))
    return dict(result=SCIP_RESULT.DELAYED)


def scip_solve(sample):
  """Solves a mip/lp using scip"""
  solver = Model()
  solver.hideOutput()
  heur = LogBestSol()
  solver.includeHeur(heur,
                     "PyHeur",
                     "custom heuristic implemented in python",
                     "Y",
                     timingmask=SCIP_HEURTIMING.BEFORENODE)
  solver.setPresolve(SCIP_PARAMSETTING.OFF)
  solver.setRealParam('limits/gap', args.gap)
  solver.readProblem(sample)
  if args.timelimit is not None:
    solver.setParam("limits/time", args.timelimit)

  solver.optimize()
  obj = float(solver.getObjVal())
  return heur.log, obj


def testRelax(idx, sample):
  log_vals, obj = scip_solve(sample)
  p = Path('/tmp') / f'server/results/{args.dataset}/{idx}.png'
  p.parent.mkdir(exist_ok=True, parents=True)
  fig, ax = plt.subplots()
  log_vals = list(filter(lambda val: val >= obj, log_vals))
  ax.plot(log_vals, label=f'Best objective {obj}')
  ax.set_ylim(obj, min(5 * obj, max(log_vals)))
  ax.legend()
  fig.savefig(p)


def load_samples(dataset: str):
  path = DATASET_PATH[dataset]
  l = []
  rng = np.random.RandomState(0)
  chosen = rng.choice(os.listdir(path), args.n_samples)
  return [Path(path) / fname for fname in chosen]


def main(argv):
  global args
  args = parser.parse_args(argv[1:])
  samples = load_samples(args.dataset)

  with Pool(len(samples)) as pool:
    pool.starmap(testRelax, enumerate(samples))


if __name__ == '__main__':
  app.run(main)
