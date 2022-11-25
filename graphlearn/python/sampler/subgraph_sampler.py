# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python.data.values import Edges, SubGraph
from graphlearn.python.utils import strategy2op
import graphlearn.python.errors as errors

class SubGraphSampler(object):
  """ Base class for sampling subgraph, 2 modes are supported:
  in_order_node and random_node.
  """

  def __init__(self,
               graph,
               seed_type,
               nbr_type,
               batch_size,
               strategy="random_node"):
    """ Create a SubGraphSampler instance.
    Args:
      graph (`Graph` object): The graph which sample from.
      seed_type (string): Sample seed type, either node type or edge type.
      nbr_type (string): Neighbor type of seeds nodes/edges.
      batch_size (int): How many nodes will be returned for `get()`.
      strategy (string, Optional): Sampling strategy, "random_node" and
        "in_order_node" are supported.
    """
    self._graph = graph
    self._seed_type = seed_type
    self._nbr_type = nbr_type
    self._batch_size = batch_size
    self._strategy = strategy
    self._client = self._graph.get_client()

    topology = self._graph.get_topology()
    self._node_type = topology.get_src_type(self._nbr_type)

  def get(self):
    """ Get sampled `SubGraph`.

    Return:
      An `SubGraph` object.
    Raise:
      `graphlearn.OutOfRangeError`
    """
    state = self._graph.node_state.get(self._seed_type)
    req = pywrap.new_subgraph_request(
        self._seed_type,
        self._nbr_type,
        strategy2op(self._strategy, "SubGraphSampler"),
        self._batch_size,
        state)
    res = pywrap.new_subgraph_response()

    status = self._client.sample_subgraph(req, res)
    if status.ok():
      node_ids = pywrap.get_node_set(res)
      row_idx = pywrap.get_row_idx(res)
      col_idx = pywrap.get_col_idx(res)
      edge_ids = pywrap.get_edge_set(res)
    else:
      if status.code() == errors.OUT_OF_RANGE:
        if self._client.connect_to_next_server():
          return self.get()
      self._graph.node_state.inc(self._seed_type)

    pywrap.del_op_response(res)
    pywrap.del_op_request(req)
    errors.raise_exception_on_not_ok_status(status)

    nodes = self._graph.get_nodes(self._node_type, node_ids)
    return SubGraph(np.stack([row_idx, col_idx], axis=0), nodes, Edges(edge_ids=edge_ids))


class RandomNodeSubGraphSampler(SubGraphSampler):
  pass


class InOrderNodeSubGraphSampler(SubGraphSampler):
  pass
