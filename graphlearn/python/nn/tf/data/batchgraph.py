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
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

from graphlearn.python.nn.data import Data
from graphlearn.python.nn.subgraph import SubGraph
from graphlearn.python.nn.tf.data.feature_handler import FeatureHandler


class BatchGraph(SubGraph):
  """A BatchGraph object, which represents a batch of `SubGraph`s.
  Nodes, edges in subgraphs are concatenated together and their offsets 
  are recorded with `graph_node_offsets` and `graph_edge_offsets`. The
  `edge_index` of subgraph is remapped according to the order offset of 
  each subgrpah and then form as a new `edge_index`.

  Args:
    edge_index: concatenated edge_index of `SubGraph`s.
    nodes: A `Data`/Tensor object denoting concatenated nodes of `SubGraph`s 
      with shape [batch_size, attr_num].
    node_schema: A (name, Decoder) tuple used to describe the nodes' feature.
    graph_node_offsets: indicates the nodes offset of each `SubGraph`.
    edges: A `Data`/Tensor object denoting concatenated edges of `SubGraph`s.
    edge_schema: A (name, Decoder) tuple used to describe the edges' feature.
    graph_edge_offsets: indicates the edges offset of each `SuGraph`.
    additional_keys: A list of keys used to indicate the additional data. Note 
      that these keys must not contain the above args. 
      Note that we require this argument in order to keep the correct order of 
      the additional data when generating Tensor format of `BatchGraph`.
  """
  def __init__(self, edge_index, nodes, node_schema, graph_node_offsets,
               edges=None, edge_schema=None, graph_edge_offsets=None, 
               additional_keys=[], **kwargs):
    super(BatchGraph, self).__init__(edge_index, nodes)
    self._edge_index = edge_index
    self._nodes = nodes
    self._node_schema = node_schema
    self._edges = edges
    self._edge_schema = edge_schema
    self._graph_node_offsets = graph_node_offsets
    self._graph_edge_offsets = graph_edge_offsets
    self._additional_keys = additional_keys
    for key, item in kwargs.items():
      self[key] = item

  @property
  def num_nodes(self):
    if isinstance(self._nodes.ids, np.ndarray):
      return self._nodes.ids.size
    else:
      return self._nodes.ids.shape.as_list()[0]

  @property
  def num_edges(self):
    if isinstance(self._edge_index, np.ndarray):
      return self._edge_index.shape[1]
    else:
      return self._edge_index.shape.as_list()[1]

  @property
  def num_graphs(self):
    """number of SubGraphs.
    """
    if isinstance(self.graph_node_offsets, np.ndarray):
      return np.amax(self.graph_node_offsets) + 1
    else:
      return tf.reduce_max(self.graph_node_offsets) + 1

  @property
  def graph_node_offsets(self):
    return self._graph_node_offsets

  @property
  def graph_edge_offsets(self):
    return self._graph_edge_offsets

  @property
  def node_schema(self):
    return self._node_schema

  @property
  def edge_schema(self):
    return self._edge_schema

  @property
  def additional_keys(self):
    return self._additional_keys

  def __getitem__(self, key):
    return getattr(self, key, None)

  def __setitem__(self, key, value):
    setattr(self, key, value)

  def transform(self, transform_func=None):
    """transforms `BatchGraph`. Default transformation is encoding 
    nodes and edges(if exist) feature to embedding.
    Args:
      transform_func: A function that takes in an `BatchGraph` object 
        and returns a transformed version. 
    """
    def transform_feat(feat, schema):
      feat_handler = FeatureHandler(schema[0], schema[1])
      return feat_handler.forward(feat)

    if self.node_schema is None:
      return self

    node = Data(self.nodes.ids, 
                self.nodes.int_attrs, 
                self.nodes.float_attrs, 
                self.nodes.string_attrs)
    node_tensor = transform_feat(node, 
      [self.node_schema[0], self.node_schema[1].feature_spec])

    edge_tensor = None
    if self.edge_schema is not None:
      edge = Data(self.edges.ids, 
                  self.edges.int_attrs, 
                  self.edges.float_attrs, 
                  self.edges.string_attrs)
      edge_tensor = transform_feat(edge, 
        [self.edge_schema[0], self.edge_schema[1].feature_spec])

    graph = BatchGraph(self.edge_index, 
                       node_tensor, self.node_schema, self.graph_node_offsets,
                       edge_tensor, self.edge_schema, self.graph_edge_offsets,
                       additional_keys=self.additional_keys)
    for key in self.additional_keys:
      graph[key] = self[key]
    return graph

  def to_graphs(self):
    """reconstructs `SubGraph`s."""
    pass
  
  def flatten(self):
    """ Flatten `BatchGraph` to numpy array in the order of
    [edge_index, nodes, graph_node_offsets, edges, graph_edge_offsets, 
    additional_keys]
    """
    def append_flatten_data(flatten_list, data) :
      data_list = [data.int_attrs, data.float_attrs, data.string_attrs, 
                   data.labels, data.weights, data.ids]
      flatten_list.extend([item for item in data_list if item is not None])
      return flatten_list

    flatten_list = [self.edge_index]
    flatten_list = append_flatten_data(flatten_list, self.nodes)
    flatten_list.append(self.graph_node_offsets)
    if self.edges is not None:
      flatten_list = append_flatten_data(flatten_list, self.edges)
      flatten_list.append(self.graph_edge_offsets)
    for key in self.additional_keys:
      flatten_list.append(self[key])
    return flatten_list
    
  @classmethod
  def from_graphs(cls, graphs, additional_keys=[]):
    """creates `BatchGraph` from a list of `SubGraph`s.
    Note that the additional data of each SubGraph must have the same
    type and all dimensions except axis=0 must be equal.

    Args:
      graphs: `SubGraph` objects.
      additional_keys: Keys(a list) of the additional data.
    """
    graph_node_offsets = BatchGraph._build_graph_node_offsets(graphs)
    graph_edge_offsets = BatchGraph._build_graph_edge_offsets(graphs)
    nodes = BatchGraph._build_data(graphs, 'node')
    edges = BatchGraph._build_data(graphs, 'edge')
    edge_index = BatchGraph._build_edge_index(graphs)
    graph = BatchGraph(edge_index, nodes, None, graph_node_offsets,
                       edges, None, graph_edge_offsets, additional_keys=additional_keys)
    for key in additional_keys:
      item = BatchGraph._build_additional_data(graphs, key)
      graph[key] = item
    return graph

  @classmethod
  def _build_graph_node_offsets(cls, graphs):
    graph_node_offsets = []
    offset = 0
    for graph in graphs:
      graph_node_offsets.append(offset)
      offset += graph.num_nodes
    return np.array(graph_node_offsets, dtype=np.int64)

  @classmethod
  def _build_graph_edge_offsets(cls, graphs):
    graph_edge_offsets = []
    offset = 0
    for graph in graphs:
      graph_edge_offsets.append(offset)
      offset += graph.num_edges
    return np.array(graph_edge_offsets, dtype=np.int64)

  @classmethod
  def _build_data(cls, graphs, name='node'):
    def np_concat(list):
      if list:
        return np.concatenate(list, axis=0)
      return None

    data_list = [[],[],[],[],[],[]]
    ele_name = ['ids', 'int_attrs', 'float_attrs', 'string_attrs', 'labels', 'weights']
    if (not name == 'node') and (graphs[0].edges is None):
      return None

    for graph in graphs:
      if name == 'node':
        item = graph.nodes
      else:
        item = graph.edges
      # flatten format.
      for i, ele in enumerate(ele_name):
        if getattr(item, ele) is not None:
          data_list[i].append(getattr(item, ele)) 
    data_list = [np_concat(x) for x in data_list]
    return Data(data_list[0], data_list[1], data_list[2], 
      data_list[3], data_list[4], data_list[5])

  @classmethod
  def _build_edge_index(cls, graphs):
    edge_index = []
    offset = 0
    for graph in graphs:
      edge_index.append(graph.edge_index + offset)
      offset += graph.num_nodes
    return np.concatenate(edge_index, axis=1)

  @classmethod
  def _build_additional_data(cls, graphs, key):
    items = []
    for graph in graphs:
      items.append(graph[key])
    return np.concatenate(items, axis=0)

  @classmethod
  def from_tensors(cls, tensors, node_schema, 
    edge_schema=None, use_edges=False, additional_keys=[], **kwargs):
    """Builds `BatchGraph` object from flatten tensors.
    The order of the build and the order of the flattening needs to be 
    strictly consistent
    Args:
      tensors: A tuple of tensors corresponding to`BatchGraph` flatten format.
      node_schema: A (name, Decoder) tuple used to describe the nodes' feature.
      use_edges: True if this graph contains edges.
      additional_keys: Keys(a list) of the additional data.
    Returns:
      A `BatchGraph` object in tensor format.
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

    edge_index = next(tensors)
    node_tensor, graph_node_offsets, edge_tensor, graph_edge_offsets = [None] * 4
    if node_schema is not None:
      node_tensor = _build_data_from_tensors(node_schema, tensors)
      graph_node_offsets = next(tensors)

    if use_edges and edge_schema is not None:
      edge_tensor = _build_data_from_tensors(edge_schema, tensors)
      graph_edge_offsets = next(tensors)

    graph = BatchGraph(edge_index, node_tensor, 
      node_schema, graph_node_offsets,
      edge_schema, graph_edge_offsets,
      additional_keys=additional_keys)
    for key in additional_keys:
      item = next(tensors)
      graph[key] = item
    return graph