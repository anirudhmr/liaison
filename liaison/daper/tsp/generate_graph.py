import numpy as np


def generate_graph(seed, num_nodes):
  r = np.random.RandomState(seed)
  # sample points from [0, 1] x [0, 1] euclidean plane.
  return r.random((num_nodes, 2))
