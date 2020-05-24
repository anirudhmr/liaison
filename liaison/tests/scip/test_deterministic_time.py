import pdb
import pickle
from multiprocessing import Pool

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from absl import app
from liaison.daper.dataset_constants import LENGTH_MAP, NORMALIZATION_CONSTANTS
from liaison.daper.milp.primitives import IntegerVariable, MIPInstance
from liaison.daper.milp.scip_mip import SCIPMIPInstance
from liaison.env.utils.rins import get_sample
from liaison.utils import ConfigDict
from pyscipopt import Model

mpl.use('Agg')
DATASET = 'milp-corlat'


def sample(i):
  seed = i
  np.random.seed(seed)
  milp = get_sample(DATASET, 'train', i % LENGTH_MAP[DATASET]['train'])
  mip = SCIPMIPInstance.fromMIPInstance(milp.mip)
  all_integer_vars = []
  feasible_ass = milp.feasible_solution
  for vname, var in mip.varname2var.items():
    if var.vtype() in ['INTEGER', 'BINARY']:
      all_integer_vars.append(vname.lstrip('t_'))

  K = min(len(all_integer_vars), np.random.randint(20, 50))
  fixed_ass = {
      all_integer_vars[i]: feasible_ass[all_integer_vars[i]]
      for i in np.random.choice(len(all_integer_vars), len(all_integer_vars) - K, replace=False)
  }
  model = mip.fix(fixed_ass)
  model.setBoolParam('randomization/permutevars', True)
  model.setIntParam('randomization/permutationseed', seed)
  model.setIntParam('randomization/randomseedshift', seed)
  model.optimize()
  solving_stats = ConfigDict(model.getSolvingStats())
  results = ConfigDict(
      solving_time=solving_stats.solvingtime,
      determinstic_time=solving_stats.deterministictime,
      nnodes=model.getNNodes(),
  )
  return results


def plt_results(results):
  X, Y = zip(*[(d.solving_time, d.determinstic_time) for d in results])
  ux, sx = np.mean(X), np.std(X)
  uy, sy = np.mean(Y), np.std(Y)
  plt.plot(X, Y)
  plt.xlabel('Solving Time')
  plt.ylabel('Deterministic Time')
  plt.xlim((ux - 2 * sx, ux + 2 * sx))
  plt.ylim((uy - 2 * sy, uy + 2 * sy))
  plt.savefig('determinstic_time.pdf')


def main(_):
  N = 1024 * 8
  with Pool(64) as pool:
    results = pool.map(sample, list(range(N)))
  results = list(sorted(results, key=lambda d: (d.solving_time, d.determinstic_time)))
  with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)
  plt_results(results)


if __name__ == '__main__':
  app.run(main)
