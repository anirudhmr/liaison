import argparse
import functools
import os
import pickle
from multiprocessing.pool import ThreadPool

from liaison.daper.tsp.dataset import TSP
from liaison.daper.tsp.generate_graph import generate_graph
from liaison.daper.tsp.solve_concorde import solve_concorde
from liaison.daper.tsp.solve_gurobi import solve_gurobi
from liaison.daper.tsp.solve_heuristic import solve_insertion

CONCORDE_TMP_DIR = '/tmp/concorde/'

parser = argparse.ArgumentParser()
parser.add_argument('--concorde_path',
                    type=str,
                    default='./tsp/concorde/concorde/TSP/concorde')
parser.add_argument('--out_file', type=str, required=True)
parser.add_argument('--num_nodes', type=int, required=True)
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()


def main():
  if not os.path.exists(args.concorde_path):
    raise Exception('Concorde not found in the specified path: %s' %
                    args.concorde_path)
  os.makedirs(CONCORDE_TMP_DIR, exist_ok=True)

  loc = generate_graph(args.seed, args.num_nodes)
  tsp = TSP()
  tsp.seed = args.seed
  tsp.locs = loc

  # adds the results to the corresponding dict.
  def f(d, l):
    obj, path, solve_time = l
    d['objective'] = obj
    d['path'] = path
    d['solve_time'] = solve_time

  res = tsp.baseline_results
  with ThreadPool(10) as pool:
    f(
        res.concorde,
        *pool.starmap(solve_concorde,
                      [(args.concorde_path, '/tmp/concorde', loc)]))

    f(res.gurobi, *pool.map(solve_gurobi, (loc, )))
    f(res.insertion_heuristics.random,
      *pool.map(functools.partial(solve_insertion, method='random'), (loc, )))
    f(res.insertion_heuristics.nearest,
      *pool.map(functools.partial(solve_insertion, method='nearest'), (loc, )))
    f(
        res.insertion_heuristics.farthest,
        *pool.map(functools.partial(solve_insertion, method='farthest'),
                  (loc, )))

  os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

  with open(args.out_file, 'wb') as f:
    pickle.dump(tsp, f)


if __name__ == '__main__':
  main()
