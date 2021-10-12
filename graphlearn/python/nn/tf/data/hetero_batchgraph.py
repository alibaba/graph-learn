# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
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

from graphlearn.python.nn.data import Data
from graphlearn.python.nn.hetero_subgraph import HeteroSubGraph
from graphlearn.python.nn.tf.data.feature_handler import FeatureHandler


class HeteroBatchGraph(HeteroSubGraph):
  """A HeteroBatchGraph object, which represents a batch of `HeteroSubGraph`s.
  Each type of Nodes, edges in subgraphs are concatenated together and their 
  offsets are recorded with `graph_node_offsets_dict` and 
  `graph_edge_offsets_dict`.

  Args:
    edge_index_dict: concatenated edge_index_dict of `HeteroSubGraph`s 
      according keys.
    nodes_dict: A dict of `Data`/Tensor object denoting concatenated nodes.
    node_schema_dict: A dict of {name: Decoder} to describe feature of each
      type of node.
    graph_node_offsets_dict: A dict, keys indicate the type of nodes, 
      values indicate the offset of each HeteroSubgraph nodes.
    edges_dict: A dict of `Data`/Tensor object denoting concatenated edges.
    edge_schema_dict: A dict of {name: Decoder} to describe feature of each
      type of edge.
    graph_edge_offsets_dict: A dict, keys indicate the type of edges, 
      values indicate the offset of each HeteroSubgraph edges.
  """
  def __init__(self, edge_index_dict, nodes_dict, node_schema_dict, 
               graph_node_offsets_dict, edges_dict=None, edge_schema_dict=None, 
               graph_edge_offsets_dict=None, **kwargs):
    super(HeteroBatchGraph, self).__init__(edge_index_dict, nodes_dict)
    self._edge_index_dict = edge_index_dict
    self._nodes_dict = nodes_dict
    self._node_schema_dict = node_schema_dict
    self._edges_dict = edges_dict
    self._edge_schema_dict = edge_schema_dict
    self._graph_node_offsets_dict = graph_node_offsets_dict
    self._graph_edge_offsets_dict = graph_edge_offsets_dict
    for key, item in kwargs.items():
      self[key] = item

  def num_nodes(self, node_type):
    nodes = self._nodes_dict[node_type]
    if isinstance(nodes.ids, np.ndarray):
      return nodes.ids.size
    else:
      return nodes.ids.shape.as_list()[0]

  def num_edges(self, edge_type):
    edge_index = self._edge_index_dict[edge_type]
    if isinstance(edge_index, np.ndarray):
      return edge_index.shape[1]
    else:
      return edge_index.shape.as_list()[1]

  @property
  def num_graphs(self):
    """number of SubGraphs.
    """
    graph_node_offsets = next(iter(self.graph_node_offsets_dict))
    if isinstance(graph_node_offsets, np.ndarray):
      return np.amax(graph_node_offsets) + 1
    else:
      return tf.reduce_max(graph_node_offsets) + 1

  @property
  def graph_node_offsets_dict(self):
    return self._graph_node_offsets_dict

  @property
  def graph_edge_offsets_dict(self):
    return self._graph_edge_offsets_dict

  @property
  def node_schema_dict(self):
    return self._node_schema_dict

  @property
  def edge_schema_dict(self):
    return self._edge_schema_dict

  def __getitem__(self, key):
    return getattr(self, key, None)

  def __setitem__(self, key, value):
    setattr(self, key, value)

  def transform(self, transform_func=None):
    """transforms `HeteroBatchGraph`. Default transformation is encoding 
    nodes feature to embedding.
    Args:
      transform_func: A function that takes in an `HeteroBatchGraph` object 
        and returns a transformed version. 
    """
    if self.node_schema_dict is None:
      return self
    node_tensor_dict = {}
    for node_type, nodes in self.nodes_dict.items():
      vertex_handler = FeatureHandler(node_type,
                                      self.node_schema_dict[node_type].feature_spec)
      node = Data(nodes.ids, 
                  nodes.int_attrs, 
                  nodes.float_attrs, 
                  nodes.string_attrs)
      node_tensor = vertex_handler.forward(node)
      node_tensor_dict[node_type] = node_tensor
    graph = HeteroBatchGraph(self.edge_index_dict, node_tensor_dict, 
                             self.node_schema_dict, self.graph_node_offsets_dict)
    return graph

  def to_graphs(self):
    """reconstructs `HeteroSubGraph`s."""
    pass
  
  def flatten(self, node_types, edge_types):
    """ return the flattened format of HeteroBatchGraph.
    Args:
      node_types: a list of node types of this HeteroBatchGraph.
      edge_types: a list of edge types of this HeteroBatchGraph.

    When we use `tf.data.Dataset.from_generator` to convert HeteroBatchGraph to
    tensor format, we must specify the types and shapes of flattened values of
    the HeteroBatchGraph. The order of the flattened data needs to be strictly 
    consistent with the order of the types and shapes.
    """
    # TODO(baole): support edges.
    flatten_list = []
    for edge_type in edge_types:
      flatten_list.append(self.edge_index_dict[edge_type])
    # nodes
    for node_type in node_types:
      nodes = self.nodes_dict.get(node_type)
      if nodes.int_attrs is not None:
        flatten_list.append(nodes.int_attrs)
      if nodes.float_attrs is not None:
        flatten_list.append(nodes.float_attrs)
      if nodes.string_attrs is not None:
        flatten_list.append(nodes.string_attrs)
      flatten_list.append(nodes.ids)
      # graph_node_offsets
      flatten_list.append(self.graph_node_offsets_dict[node_type])
    return flatten_list
    
  @classmethod
  def from_graphs(cls, graphs):
    """creates `HeteroBatchGraph` from a list of `HeteroSubGraph`s.

    Args:
      graphs: `HeteroSubGraph` objects.
    """
    graph_node_offsets_dict = HeteroBatchGraph._build_graph_node_offsets(graphs)
    graph_edge_offsets_dict = HeteroBatchGraph._build_graph_edge_offsets(graphs)
    nodes_dict = HeteroBatchGraph._build_data(graphs, 'node')
    edges_dict = HeteroBatchGraph._build_data(graphs, 'edge')
    edge_index_dict = HeteroBatchGraph._build_edge_index(graphs)
    graph = HeteroBatchGraph(edge_index_dict, 
                             nodes_dict, None, graph_node_offsets_dict,
                             edges_dict, None, graph_edge_offsets_dict)
    return graph

  @classmethod
  def _build_graph_node_offsets(cls, graphs):
    offsets_dict = {}
    for node_type in graphs[0].node_types:
      offsets = []
      offset = 0
      for graph in graphs:
        offsets.append(offset)
        offset += graph.num_nodes(node_type)
      offsets_dict[node_type] = np.array(offsets, dtype=np.int64)
    return offsets_dict

  @classmethod
  def _build_graph_edge_offsets(cls, graphs):
    offsets_dict = {}
    for edge_type in graphs[0].edge_types:
      offsets = []
      offset = 0
      for graph in graphs:
        offsets.append(offset)
        offset += graph.num_edges(edge_type)
      offsets_dict[edge_type] = np.array(offsets, dtype=np.int64)
    return offsets_dict

  @classmethod
  def _build_data(cls, graphs, data_type='node'):
    def list_append(list, item):
      if item is not None:
        list.append(item)
      return list

    def np_concat(list):
      if list:
        return np.concatenate(list, axis=0)
      return None

    #TODO(baole): support other labels and weights.
    if data_type != 'node':
      return None

    data_dict = {}
    for node_type in graphs[0].node_types:
      ids_list = []
      int_attrs_list = []
      float_attrs_list = []
      string_attrs_list = []
      for graph in graphs:
        item = graph.nodes_dict[node_type]
        # flatten format.
        ids_list = list_append(ids_list, item.ids)
        int_attrs_list = list_append(int_attrs_list, item.int_attrs)
        float_attrs_list = list_append(float_attrs_list, item.float_attrs)
        string_attrs_list = list_append(string_attrs_list, item.string_attrs)
      ids = np_concat(ids_list)
      ints = np_concat(int_attrs_list)
      floats = np_concat(float_attrs_list)
      strings = np_concat(string_attrs_list)
      data_dict[node_type] = Data(ids, ints, floats, strings)
    return data_dict

  @classmethod
  def _build_edge_index(cls, graphs):
    edge_index_dict = {}
    for edge_type in graphs[0].edge_types:
      edge_index = []
      src_offset, dst_offset = 0, 0
      for graph in graphs:
        edge_index.append(np.array([graph.edge_index_dict[edge_type][0] + src_offset,
                                    graph.edge_index_dict[edge_type][1] + dst_offset]))
        src_offset += graph.num_nodes(edge_type[0])
        dst_offset += graph.num_nodes(edge_type[2])
      edge_index_dict[edge_type] = np.concatenate(edge_index, axis=1)
    return edge_index_dict

  @classmethod
  def from_tensors(cls, tensors, node_schema, edge_schema, **kwargs):
    """builds `HeteroBatchGraph` object from flatten tensors.
    Args:
      tensors: A tuple of tensors corresponding to`HeteroBatchGraph` flatten format.
      node_schema: A list of (name, Decoder) tuple used to describe the nodes' feature.
      edge_schema: A list of (name, Decoder) tuple used to describe the edges' feature.
    Returns:
      A `HeteroBatchGraph` object in tensor format.

    Note that the order of the parsed results must be the same as the order of the previous 
    `flatten` function, so here we use node_schema list format instead of dict format.
    """
    cursor = [0]
    def next(tensors):
      t = tensors[cursor[0]]
      cursor[0] += 1
      return t

    def build_node_from_tensors(feature_schema, tensors):
      """Constructs nodes `Data` in Tensor format.
      Args:
        feature_schema: A (name, Decoder) tuple used to parse the feature.
      Returns:
        A `Data` object in Tensor format.
      """
      if feature_schema[1].int_attr_num > 0:
        int_attrs = next(tensors)
      else:
        int_attrs = None
      if feature_schema[1].float_attr_num > 0:
        float_attrs = next(tensors)
      else:
        float_attrs = None
      if feature_schema[1].string_attr_num > 0:
        string_attrs = next(tensors)
      else:
        string_attrs = None
      ids = next(tensors)
      feature_tensor = Data(ids,
                            ints=int_attrs,
                            floats=float_attrs,
                            strings=string_attrs)
      return feature_tensor

    edge_index_dict = {}
    node_tensor_dict = {}
    graph_node_offsets_dict = {}
    node_schema_dict, edge_schema_dict = {}, {}
    for item in edge_schema:
      edge_index_dict[item[0]] = next(tensors)
      edge_schema_dict[item[0]] = item[1]
    for item in node_schema:
      node_tensor_dict[item[0]] = build_node_from_tensors(item, tensors)
      graph_node_offsets_dict[item[0]] = next(tensors)
      node_schema_dict[item[0]] = item[1]

    #TODO(baole): support Edge.
    graph = HeteroBatchGraph(edge_index_dict, node_tensor_dict, 
      node_schema_dict, graph_node_offsets_dict)
    return graph