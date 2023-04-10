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
               nbr_type,
               num_nbrs=[0],
               need_dist=False):
    """ Create a SubGraphSampler instance.
    Args:
      graph (`Graph` object): The graph which sample from.
      nbr_type (string): Neighbor type of seeds nodes/edges.
      strategy (string, Optional): Sampling strategy, "random_node/edge" and
        "in_order_node/edge" are supported.
      num_nbrs (List[int], Optional): number of neighbors for each hop.
      need_dist: Whether need return the distance from each node in subgraph
        to src and dst. Note that this arg is valid only when `dst_ids` in
        `get()` is not None and size of `dst_ids` is 1.
    """
    self._graph = graph
    self._nbr_type = nbr_type
    self._num_nbrs = num_nbrs
    self._need_dist = need_dist
    self._client = self._graph.get_client()

    topology = self._graph.get_topology()
    self._node_type = topology.get_src_type(self._nbr_type)

  def get(self, ids, dst_ids=None):
    """ Get sampled `SubGraph`.
    Args:
      ids: A 1d numpy array, the input ids(either src_id, or [src_id, dst_id]),
        type=np.int64.

    Return:
      An `SubGraph` object.
    Raise:
      `graphlearn.OutOfRangeError`
    """
    state = self._graph.node_state.get(self._seed_type)
    req = pywrap.new_subgraph_request(
        self._seed_type,
        self._nbr_type,
        self._num_nbrs,
        self._need_dist)
    pywrap.set_subgraph_request(req, ids, dst_ids)
    res = pywrap.new_subgraph_response()

    status = self._client.sample_subgraph(req, res)
    if status.ok():
      node_ids = pywrap.get_node_set(res)
      row_idx = pywrap.get_row_idx(res)
      col_idx = pywrap.get_col_idx(res)
      edge_ids = pywrap.get_edge_set(res)
      dist_to_src, dist_to_dst = None, None
      if self._need_dist:
        dist_to_src = pywrap.get_dist_to_src(res)
        dist_to_dst = pywrap.get_dist_to_dst(res)
    else:
      if status.code() == errors.OUT_OF_RANGE:
        if self._client.connect_to_next_server():
          return self.get()
      self._graph.node_state.inc(self._seed_type)

    pywrap.del_op_response(res)
    pywrap.del_op_request(req)
    errors.raise_exception_on_not_ok_status(status)

    nodes = self._graph.get_nodes(self._node_type, node_ids)
    subgraph = SubGraph(np.stack([row_idx, col_idx], axis=0),
                        nodes,
                        Edges(edge_ids=edge_ids))
    subgraph.dist_to_src = dist_to_src
    subgraph.dist_to_dst = dist_to_dst
    return subgraph
