import argparse
import os
import pickle
from pathlib import Path

from liaison.daper.dataset_constants import DATASET_PATH
from liaison.daper.milp.dataset import MILP, MILP_PRIMITIVE
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', required=True)
parser.add_argument('--out_dir', '-o', required=True)
args = parser.parse_args()


def load_graph(fname):
  with open(fname, 'rb') as f:
    return pickle.load(f)


def convert_graph(graph):
  milp = MILP_PRIMITIVE()
  for k, v in milp.items():
    if isinstance(v, dict):
      for k2 in v:
        if k2 in graph[k]:
          milp[k][k2] = graph[k][k2]
    else:
      if k in graph:
        milp[k] = graph[k]
  return milp


def write_graph(graph, fname):
  Path(fname).parent.mkdir(parents=True, exist_ok=True)
  with open(fname, 'wb') as f:
    pickle.dump(graph, f)


def main():
  for dtype in ['train', 'valid', 'test']:
    d = f'{DATASET_PATH[args.dataset]}/{dtype}'
    for fname in tqdm(os.listdir(d)):
      graph = load_graph(f'{d}/{fname}')
      graph = convert_graph(graph)
      write_graph(graph, f'{args.out_dir}/{dtype}/{fname}')


if __name__ == '__main__':
  main()
