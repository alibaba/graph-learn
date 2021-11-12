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
import tensorflow as tf

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
      type of edge. The edges_dict is not None when Decoder is not None.
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
    def transform_feat(feat, schema):
      feat_handler = FeatureHandler(schema[0], schema[1])
      return feat_handler.forward(feat)

    if self.node_schema_dict is None:
      return self

    node_tensor_dict = {}
    for node_type, nodes in self.nodes_dict.items():
      node = Data(nodes.ids, 
                  nodes.int_attrs, 
                  nodes.float_attrs, 
                  nodes.string_attrs)
      node_tensor = transform_feat(node, 
        [node_type, self.node_schema_dict[node_type].feature_spec])
      node_tensor_dict[node_type] = node_tensor
    
    edge_tensor_dict = {}
    for edge_type, edges in self.edges_dict.items():
      edge = Data(edges.ids, 
                  edges.int_attrs, 
                  edges.float_attrs, 
                  edges.string_attrs)
      edge_tensor = transform_feat(edge, 
        [edge_type, self.edge_schema_dict[edge_type].feature_spec])
      edge_tensor_dict[edge_type] = edge_tensor    

    graph = HeteroBatchGraph(self.edge_index_dict, 
                             node_tensor_dict, self.node_schema_dict, 
                             self.graph_node_offsets_dict,
                             edge_tensor_dict, self.edge_schema_dict,
                             self.graph_edge_offsets_dict)
    return graph

  def to_graphs(self):
    """reconstructs `HeteroSubGraph`s."""
    pass
  
  def flatten(self, node_types, edge_types, use_edges=False):
    """ return the flattened format of HeteroBatchGraph.
    Flatten `HeteroBatchGraph` to numpy array in the order of
    [edge_index_dict, nodes_dict, graph_node_offsets_dict, 
     edges_dict, graph_edge_offsets_dict].

    Args:
      node_types: a list of node types of this HeteroBatchGraph.
      edge_types: a list of edge types of this HeteroBatchGraph.
      use_edges: True if this graph contains edges.

    When we use `tf.data.Dataset.from_generator` to convert HeteroBatchGraph to
    tensor format, we must specify the types and shapes of flattened values of
    the HeteroBatchGraph. The order of the flattened data needs to be strictly 
    consistent with the order of the types and shapes.
    """
    def append_flatten_data(flatten_list, data) :
      data_list = [data.int_attrs, data.float_attrs, data.string_attrs, 
                   data.labels, data.weights, data.ids]
      flatten_list.extend([item for item in data_list if item is not None])
      return flatten_list

    flatten_list = []
    for edge_type in edge_types:
      flatten_list.append(self.edge_index_dict[edge_type])
    # nodes
    for node_type in node_types:
      nodes = self.nodes_dict.get(node_type)
      flatten_list = append_flatten_data(flatten_list, nodes)
      # graph_node_offsets
      flatten_list.append(self.graph_node_offsets_dict[node_type])
    # edges
    if use_edges:
      for edge_type in edge_types:
        edges = self.edges_dict.get(edge_type)
        flatten_list = append_flatten_data(flatten_list, edges)
        # graph_edge_offsets
        flatten_list.append(self.graph_edge_offsets_dict[node_type])
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
  def _build_data(cls, graphs, name='node'):
    def np_concat(list):
      if list:
        return np.concatenate(list, axis=0)
      return None
    
    if name == 'node':
      types = graphs[0].node_types
    else:
      types = graphs[0].edge_types
      if not graphs[0].edges_dict:
        return None

    data_dict = {}
    for data_type in types:
      data_list = [[],[],[],[],[],[]]
      ele_name = ['ids', 'int_attrs', 'float_attrs', 'string_attrs', 'labels', 'weights']
      for graph in graphs:
        if name == 'node':
          item = graph.nodes_dict[data_type]
        else:
          item = graph.edges_dict[data_type]
        # flatten format.
        for i, ele in enumerate(ele_name):
          if getattr(item, ele) is not None:
            data_list[i].append(getattr(item, ele)) 
      data_list = [np_concat(x) for x in data_list]
      data_dict[data_type] = Data(data_list[0], data_list[1], data_list[2], 
        data_list[3], data_list[4], data_list[5])
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
  def from_tensors(cls, tensors, node_schema, edge_schema, use_edges=False, **kwargs):
    """builds `HeteroBatchGraph` object from flatten tensors.
    Args:
      tensors: A tuple of tensors corresponding to`HeteroBatchGraph` flatten format.
      node_schema: A list of (name, Decoder) tuple used to describe the nodes' feature.
      edge_schema: A list of (name, Decoder) tuple used to describe the edges' feature.
      use_edges: True if this graph contains edges.
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

    def _build_data_from_tensors(feature_schema, tensors):
      """Constructs nodes `Data` in Tensor format.
      Args:
        feature_schema: A (name, Decoder) tuple used to parse the feature.
      Returns:
        A `Data` object in Tensor format.
      """
      int_attrs, float_attrs, string_attrs, labels, weights, ids = [None]*6
      if feature_schema[1].int_attr_num > 0:
        int_attrs = next(tensors)
      if feature_schema[1].float_attr_num > 0:
        float_attrs = next(tensors)
      if feature_schema[1].string_attr_num > 0:
        string_attrs = next(tensors)
      if feature_schema[1].labeled:
        labels = next(tensors)
      if feature_schema[1].weighted:
        weights = next(tensors)
      ids = next(tensors)
      feature_tensor = Data(ids,
                            ints=int_attrs,
                            floats=float_attrs,
                            strings=string_attrs,
                            labels=labels,
                            weights=weights)
      return feature_tensor

    edge_index_dict, node_tensor_dict, graph_node_offsets_dict, \
    node_schema_dict, edge_tensor_dict, graph_edge_offsets_dict, \
    edge_schema_dict = {},{},{},{},{},{},{}
    for item in edge_schema:
      edge_index_dict[item[0]] = next(tensors)
    for item in node_schema:
      node_tensor_dict[item[0]] = _build_data_from_tensors(item, tensors)
      graph_node_offsets_dict[item[0]] = next(tensors)
      node_schema_dict[item[0]] = item[1]
    if use_edges:
      for item in edge_schema:
        edge_tensor_dict[item[0]] = _build_data_from_tensors(item, tensors)
        graph_edge_offsets_dict[item[0]] = next(tensors)
        edge_schema_dict[item[0]] = item[1]

    graph = HeteroBatchGraph(edge_index_dict, 
      node_tensor_dict, node_schema_dict, graph_node_offsets_dict,
      edge_tensor_dict, edge_schema_dict, graph_edge_offsets_dict)
    return graph