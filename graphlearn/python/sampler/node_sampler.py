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

from graphlearn import pywrap_graphlearn as pywrap
import graphlearn.python.errors as errors
import graphlearn.python.utils as utils


class NodeSampler(object):
  """ Sampling a batch of nodes from graph, 3 modes are supported:
  by_order, random and shuffle.
  """

  def __init__(self,
               graph,
               t,
               batch_size,
               strategy="by_order",
               node_from=pywrap.NodeFrom.NODE,
               mask=utils.Mask.NONE):
    """ Create a Base NodeSampler..

    Args:
      graph (`Graph` object): The graph which sample from.
      t (string): type of node or egde. If t is a type of node, then
        `NodeSampler` will sample from node source. Else if `t` is a type
        of edge, then `node_from=EDGE_SRC` indicates that the nodes will
        be sampled from edges's source nodes, `node_from=EDGE_DST`
        indicates that the nodes will be sampled from edges's dst nodes.
      batch_size (int): How many nodes will be returned for `get()`.
      strategy (string, Optional): Sampling strategy. "by_order", "random"
        and "shuffle" are supported.
        "by_order": get nodes by the order of how the specified node is stored,
          if all the specified type of nodes are visited,
          `graphlearn.OutOfRangeError` will be raised.
          NodeSamplers that process the same node will share the same state.
        "random": randomly visit nodes, no state will be kept.
        "shuffle": visit the nodes with shuffling, if all the specified type of
          nodes are visited, `graphlearn.OutOfRangeError` will be raised.
          NodeSamplers that process the same node will share the same state.
     node_from (graphlearn.NODE | graphlearn.EDGE_SRC | graphlearn.EDGE_DST):
        `graphlearn.NODE`: get node from node data, and `t` must be a node type.
        `graphlearn.EDGE_SRC`: get node from source node of edge data, and `t`
          must be an edge type.
        `graphlearn.EDGE_DST`: get node from destination node of edge data, and
          `t` must be an edge type.
    """
    self._graph = graph
    self._type = t
    self._batch_size = batch_size
    self._strategy = strategy
    self._client = self._graph.get_client()
    self._node_from = node_from
    self._mask = mask

    if self._node_from == pywrap.NodeFrom.NODE:
      if self._type not in self._graph.get_node_decoders().keys():
        raise ValueError('Graph has no node type of {}'.format(self._type))
      self._node_type = self._type
    else:
      topology = self._graph.get_topology()
      src_type = topology.get_src_type(self._type)
      dst_type = topology.get_dst_type(self._type)
      self._src_type, self._dst_type = src_type, dst_type
      if self._node_from == pywrap.NodeFrom.EDGE_SRC:
        self._node_type = src_type
      else:
        self._node_type = dst_type

  def get(self):
    """ Get batched sampled nodes.

    Return:
      A `Nodes` object, shape=[`batch_size`]
    Raise:
      `graphlearn.OutOfRangeError`
    """
    mask_type = utils.get_mask_type(self._type, self._mask)
    state = self._graph.node_state.get(mask_type)
    req = pywrap.new_get_nodes_request(mask_type,
                                       self._strategy,
                                       self._node_from,
                                       self._batch_size,
                                       state)

    res = pywrap.new_get_nodes_response()
    status = self._client.get_nodes(req, res)
    if not status.ok():
      self._graph.node_state.inc(mask_type)
    else:
      ids = pywrap.get_node_ids(res)

    pywrap.del_op_response(res)
    pywrap.del_op_request(req)
    errors.raise_exception_on_not_ok_status(status)

    nodes = self._graph.get_nodes(self._node_type, ids)
    return nodes


class RandomNodeSampler(NodeSampler):
  pass


class ByOrderNodeSampler(NodeSampler):
  pass


class ShuffleNodeSampler(NodeSampler):
  pass
