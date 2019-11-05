import argparse
import copy
import os
import sys
from threading import Thread

import liaison.utils as U
import networkx as nx
import numpy as np
import queue
import tensorflow as tf
from absl import app, logging
from absl.testing import absltest
from caraml.zmq import DataFetcher, ZmqServer
from liaison.agents import StepOutput
from liaison.agents.gcn import Agent as GCNAgent
from liaison.distributed.exp_serializer import get_deserializer, get_serializer
from liaison.distributed.trajectory import Trajectory
# use this to get large graph.
from liaison.env.utils.shortest_path import generate_networkx_graph
from liaison.specs.specs import BoundedArraySpec
from liaison.utils import ConfigDict
from tensorflow.contrib.framework import nest
from tensorflow.python.client import timeline

N_NODES = 3
SEED = 42

REMOTE_PORT = 15032
WORKER_COMM_PORT = 15033

# set after parsing flags
B = 64
T = 128


class VtraceAgentTest:

  def _large_graph(self):
    # generate graph with 32 nodes.
    nx_graph, path = generate_networkx_graph(SEED, [32, 33])
    src_node, target_node = path[0], path[-1]
    nx_graph = nx_graph.to_directed()
    # Create graph features from the networkx graph
    # make sure to set all the static fields in the created features.
    # also initialize the dynamic fields to the right values.
    nodes = np.zeros((len(nx_graph), 6), dtype=np.float32)

    edges = np.zeros([nx_graph.number_of_edges(), 2], dtype=np.float32)
    weights = nx.get_edge_attributes(nx_graph, 'distance')
    for i, edge in enumerate(nx_graph.edges()):
      edges[i][0] = weights[edge]

    senders, receivers = zip(*nx_graph.edges())
    graph = dict(nodes=nodes,
                 edges=edges,
                 globals=np.zeros(1, dtype=np.float32),
                 senders=np.array(senders, dtype=np.int32),
                 receivers=np.array(receivers, dtype=np.int32),
                 n_node=np.array(len(nodes), dtype=np.int32),
                 n_edge=np.array(len(edges), dtype=np.int32))
    return graph

  def _get_graph_features(self):
    # get timestep stacked and batched graph features
    def f(*l):
      return np.stack(l, axis=0)

    graph_features = self._large_graph()
    graph_features = nest.map_structure(f, *[graph_features] * B)
    # return nest.map_structure(f, *[graph_features] * (T + 1))
    return graph_features

  def _start_remote_server(self):
    server = ZmqServer(host='*',
                       port=REMOTE_PORT,
                       serializer=get_serializer(),
                       deserializer=get_deserializer(),
                       auth=False)
    graph = self._get_graph_features()

    def handler(_):
      return [graph]

    thr = Thread(target=lambda: server.start_loop(handler, blocking=True))
    thr.daemon = True
    thr.start()
    return thr

  def _request_generator(self, n_requests):
    for i in range(n_requests):
      yield i
    return

  def _traj_spec(self):

    def expand_spec(spec):
      spec = copy.deepcopy(spec)
      np.expand_dims(spec, axis=0)
      return spec

    return nest.map_structure(expand_spec, self._get_graph_features())

  def testProfiler(self):
    self._start_remote_server()
    N_ITERS = 100
    request_generator = self._request_generator(N_ITERS * (T + 1))

    q = queue.Queue(maxsize=1000)
    fetcher = DataFetcher(handler=lambda _, data: q.put(data, block=True),
                          remote_host='127.0.0.1',
                          remote_port=REMOTE_PORT,
                          requests=request_generator,
                          worker_comm_port=WORKER_COMM_PORT,
                          remote_serializer=get_serializer(),
                          remote_deserialzer=get_deserializer(),
                          n_workers=1,
                          threads_per_worker=8,
                          worker_handler=None,
                          auth=False)
    traj_spec = self._traj_spec()
    thr = fetcher.start()
    for i in range(N_ITERS):
      l = []
      while len(l) < T + 1:
        l.extend(q.get().data)
      l = Trajectory.batch(l, traj_spec)

    fetcher.join()
    print('Done!')


def main(_):
  obj = VtraceAgentTest()
  obj.testProfiler()


if __name__ == '__main__':
  app.run(main)
