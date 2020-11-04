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
""" Sample Edges from Graph, supports by_order, random and shuffle.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python.errors import raise_exception_on_not_ok_status


class EdgeSampler(object):
  """ Base class for sampling edges from graph.
  """

  def __init__(self,
               graph,
               edge_type,
               batch_size,
               strategy="by_order"):
    """ Create a Base EdgeSampler instance.
    Args:
      graph (`Graph` object): The graph which sample from.
      edge_type (string): Sample edges of the specified edge_type.
      batch_size (int): How many edges will be returned for `get()`.
      strategy (string, Optional): Sampling strategy. "by_order", "random"
        and "shuffle" are supported.
        "by_order": get edges by order of how the specified edge is stored,
          if the specified type of edges are totally visited,
          `graphlearn.OutOfRangeError` will be raised. Several
          `EdgeSampler`s with same type will hold a single state.
        "random": randomly visit edges, no state will be kept.
        "shuffle": visit the edges with shuffling, if the specified type of
          edges are totally visited, `graphlearn.OutOfRangeError` will be
          raised. Several `EdgeSampler`s with same type will hold a single
          state.
    """
    self._graph = graph
    self._edge_type = edge_type
    self._batch_size = batch_size
    self._strategy = strategy
    self._client = self._graph.get_client()

    topology = self._graph.get_topology()
    self._node_decoders = self._graph.get_node_decoders()
    self._edge_decoders = self._graph.get_edge_decoders()

    self._src_type, self._dst_type = \
      topology.get_src_type(edge_type), topology.get_dst_type(edge_type)

  def get(self):
    """ Get batched sampled `Edges`.

    Return:
      An `Edges` object, shape=[`batch_size`]
    Raise:
      `graphlearn.OutOfRangeError`
    """
    state = self._graph.edge_state.get(self._edge_type)
    req = pywrap.new_get_edges_request(
        self._edge_type, self._strategy, self._batch_size, state)
    res = pywrap.new_get_edges_response()

    status = self._client.get_edges(req, res)
    if not status.ok():
      if self._client.connect_to_next_server():
        req = pywrap.new_get_edges_request(
          self._edge_type, self._strategy, self._batch_size, state)
        res = pywrap.new_get_edges_response()
        status = self._client.get_edges(req, res)
        src_ids = pywrap.get_edge_src_id(res)
        dst_ids = pywrap.get_edge_dst_id(res)
        edge_ids = pywrap.get_edge_id(res)
      else:
        self._graph.edge_state.inc(self._edge_type)
    else:
      src_ids = pywrap.get_edge_src_id(res)
      dst_ids = pywrap.get_edge_dst_id(res)
      edge_ids = pywrap.get_edge_id(res)

    pywrap.del_op_response(res)
    pywrap.del_op_request(req)
    raise_exception_on_not_ok_status(status)

    edges = self._graph.get_edges(self._edge_type,
                                  src_ids,
                                  dst_ids)
    edges.edge_ids = edge_ids
    return edges


class RandomEdgeSampler(EdgeSampler):
  pass


class ByOrderEdgeSampler(EdgeSampler):
  pass


class ShuffleEdgeSampler(EdgeSampler):
  pass
