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
""" Classes for different neighbor samplers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python.errors import raise_exception_on_not_ok_status
from graphlearn.python.values import Layer, Layers
from graphlearn.python.utils import strategy2op


class NeighborSampler(object):
  """ Sampling neighbors of given ids and edges between
  ids and neighbors from graph.
  """

  def __init__(self,
               graph,
               meta_path,
               expand_factor,
               strategy="random"):
    """ Create a Base NeighborSampler instance.

    Args:
      graph (`Graph` object): The graph which sample from.
      meta_path (list): a list of string of edge_type.
      expand_factor (int | list): Number of neighbors
        sampled from all the in nodes of the node.
        int: indicates the number of neighbors sampled for each node.
        list of int: indicates the number of neighbors sampled for
          each node of each hop.
      strategy: A string, sampling strategy. "random", "in_degree",
        or "edge_weight".
    """
    self._graph = graph
    if isinstance(meta_path, list):
      self._meta_path = meta_path
    elif isinstance(meta_path, tuple):
      self._meta_path = list(meta_path)
    else:
      self._meta_path = [meta_path]

    if isinstance(expand_factor, int):
      self._expand_factor = [expand_factor]
    elif isinstance(expand_factor, list):
      self._expand_factor = expand_factor
    elif isinstance(expand_factor, tuple):
      self._expand_factor = list(expand_factor)
    else:
      raise ValueError("`expand_factor` must be int or list of int.")

    self._strategy = [strategy] * len(self._expand_factor)
    self._client = self._graph.get_client()

    self._src_types = []
    self._dst_types = []

    topology = self._graph.get_topology()
    self._node_decoders = self._graph.get_node_decoders()
    self._edge_decoders = self._graph.get_edge_decoders()

    for edge_type in self._meta_path:
      self._src_types.append(topology.get_src_type(edge_type))
      self._dst_types.append(topology.get_dst_type(edge_type))

  def get(self, ids):  # pylint: disable=unused-argument
    """ Get batched samples.

    Args:
      ids: A 1d numpy array, the input ids whose neighbors will be returned,
        type=np.int64.

    Return:
      A `Layers` object.
    """

    if len(self._meta_path) != len(self._expand_factor):
      raise ValueError("The meta_path must have the same number"
                       "of elements as num_at_each_hop")

    src_ids = ids
    current_batch_size = ids.size
    layers = Layers()
    for i in range(len(self._meta_path)):
      req = self._make_req(i, src_ids)
      res = pywrap.new_sampling_response()
      status = self._client.sample_neighbor(req, res)
      if status.ok():
        nbr_ids = pywrap.get_sampling_node_ids(res)
        edge_ids = pywrap.get_sampling_edge_ids(res)

      pywrap.del_op_response(res)
      pywrap.del_op_request(req)
      raise_exception_on_not_ok_status(status)

      dst_type = self._dst_types[i]
      layer_nodes = self._graph.get_nodes(dst_type,
                                          nbr_ids,
                                          shape=(current_batch_size,
                                                 self._expand_factor[i]))

      ids = src_ids.repeat(self._expand_factor[i]).flatten()
      nbr_ids_flat = nbr_ids.flatten()
      layer_edges = \
        self._graph.get_edges(self._meta_path[i],
                              ids,
                              nbr_ids_flat,
                              shape=(current_batch_size,
                                     self._expand_factor[i]))
      layer_edges.edge_ids = edge_ids
      layers.append_layer(Layer(layer_nodes, layer_edges))
      current_batch_size = nbr_ids_flat.size

      src_ids = nbr_ids
    return layers

  def _make_req(self, index, src_ids):
    """ Make rpc request.
    """
    if len(self._meta_path) > len(self._expand_factor):
      raise ValueError("input too many meta_path, and can not decide each hops")

    sampler = strategy2op(self._strategy[index], "Sampler")
    req = pywrap.new_sampling_request(
        self._meta_path[index], sampler, self._expand_factor[index])
    pywrap.set_sampling_request(req, src_ids)
    return req


class RandomNeighborSampler(NeighborSampler):
  pass

class RandomWithoutReplacementNeighborSampler(NeighborSampler):
  pass

class EdgeWeightNeighborSampler(NeighborSampler):
  pass


class TopkNeighborSampler(NeighborSampler):
  pass


class InDegreeNeighborSampler(NeighborSampler):
  pass


class FullNeighborSampler(NeighborSampler):
  """ Get all the neighbors of given node ids.
  return Layer of SparseNodes and SparseEdges.
  """
  def get(self, ids):  # pylint: disable=unused-argument
    if len(self._meta_path) != len(self._expand_factor):
      raise ValueError("The meta_path must have the same number"
                       "of elements as num_at_each_hop")

    ids = ids.flatten()
    src_ids = ids
    current_batch_size = ids.size
    layers = Layers()
    for i in range(len(self._meta_path)):
      # req, res & call method.
      req = self._make_req(i, src_ids)
      res = pywrap.new_sampling_response()
      status = self._client.sample_neighbor(req, res)
      if status.ok():
        src_degrees = pywrap.get_sampling_node_degrees(res)
        dense_shape = (current_batch_size, max(src_degrees))

        nbr_ids = pywrap.get_sampling_node_ids(res)
        edge_ids = pywrap.get_sampling_edge_ids(res)

      pywrap.del_op_response(res)
      pywrap.del_op_request(req)
      raise_exception_on_not_ok_status(status)

      dst_type = self._dst_types[i]
      layer_nodes = self._graph.get_nodes(dst_type,
                                          nbr_ids,
                                          offsets=src_degrees,
                                          shape=dense_shape)

      ids = np.concatenate([src_ids[idx].repeat(d) for \
                              idx, d in enumerate(src_degrees)])
      nbr_ids_flat = nbr_ids.flatten()
      layer_edges = \
        self._graph.get_edges(self._meta_path[i],
                              ids,
                              nbr_ids_flat,
                              offsets=src_degrees,
                              shape=dense_shape)
      layer_edges.edge_ids = edge_ids
      layers.append_layer(Layer(layer_nodes, layer_edges))
      current_batch_size = nbr_ids_flat.size

      src_ids = nbr_ids
    return layers
