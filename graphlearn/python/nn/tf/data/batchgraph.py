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
import numpy as np
import tensorflow as tf

from graphlearn.python.nn.tf.data.entity import Vertex
from graphlearn.python.nn.tf.data.subgraph import SubGraph
from graphlearn.python.nn.tf.data.feature_group import FeatureHandler


class BatchGraph(SubGraph):
  def __init__(self, edge_index, nodes, node_schema, node_graph_id,
               edges=None, edge_schema=None, edge_graph_id=None, 
               additional_keys=[], **kwargs):
    """Creates a BatchGraph object, which represents a batch of SubGraphs.

    Args:
      edge_index: concatenated edge_index of SubGraphs.
      nodes: concatenated nodes of SubGraphs with shape [batch_size, attr_num].
      node_schema: A (name, Decoder) tuple used to describe the nodes' feature.
      node_graph_id: indicates which SubGraph the node belong to.
      edges: concatenated edges of SubGraphs.
      node_schema: A (name, Decoder) tuple used to describe the edges' feature.
      edge_graph_id: indicates which SubGraph the edge belong to.
      additional_keys: A list of keys used to indicate the additional data. Note 
        that these keys must not contain the above args.
    """
    super(BatchGraph, self).__init__(edge_index, nodes)
    self._edge_index = edge_index
    self._nodes = nodes
    self._node_schema = node_schema
    self._edges = edges
    self._edge_schema = edge_schema
    self._node_graph_id = node_graph_id
    self._edge_graph_id = edge_graph_id
    self._additional_keys = additional_keys
    for key, item in kwargs.items():
      self[key] = item

  @property
  def num_graphs(self):
    """number of SubGraphs.
    """
    if isinstance(self.node_graph_id, np.ndarray):
      return np.amax(self.node_graph_id) + 1
    else:
      return tf.reduce_max(self.node_graph_index) + 1

  @property
  def node_graph_id(self):
    """ indicates which SubGraph the node belong to.
    """
    return self._node_graph_id

  @property
  def edge_graph_id(self):
    """ indicates which SubGraph the edge belong to.
    """
    return self._edge_graph_id

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

  def forward(self, shared_param=None):
    """feature encoding function, which returns
    encoded feature embedding as nodes with shape [batch_size, dim].
    """

    def tf_transpose(item):
      if item is not None:
        item = tf.transpose(item)
      return item

    if self.node_schema is None:
      return self
    vertex_handler = FeatureHandler(self.node_schema[0],
                                    self.node_schema[1].feature_spec)
    ints, floats, strings =\
      tf_transpose(self.nodes.int_attrs),\
      tf_transpose(self.nodes.float_attrs),\
      tf_transpose(self.nodes.string_attrs)
    node = Vertex(self.nodes.ids, ints, floats, strings)
    node_tensor = vertex_handler.forward(node)
    graph = BatchGraph(self.edge_index, node_tensor, 
                       self.node_schema, self.node_graph_id,
                       additional_keys=self.additional_keys)
    for key in self.additional_keys:
      graph[key] = self[key]
    return graph

  def to_graphs(self):
    """reconstructs subgraphs."""
    pass
  
  def flatten(self):
    flatten_list = []
    flatten_list.append(self.edge_index)
    nodes = self.nodes
    # nodes
    flatten_list.append(nodes.ids)
    if nodes.int_attrs is not None:
      flatten_list.append(nodes.int_attrs)
    if nodes.float_attrs is not None:
      flatten_list.append(nodes.float_attrs)
    if nodes.string_attrs is not None:
      flatten_list.append(nodes.string_attrs)
    # node_graph_id
    flatten_list.append(self.node_graph_id)
    # additional data.
    for key in self.additional_keys:
      flatten_list.append(self[key])
    return flatten_list
    
  @classmethod
  def from_graphs(cls, graphs, additional_keys=[]):
    """create BatchGraph for a list of SubGraphs.
    Note that the additional data of each SubGraph must have the same
    type and all dimensions except axis=0 must be equal.

    Args:
      graphs: SubGraph objects.
      additional_keys: Keys(a list) of the additional data.
    """
    node_graph_id = BatchGraph._build_node_graph_id(graphs)
    edge_graph_id = BatchGraph._build_edge_graph_id(graphs)
    nodes = BatchGraph._build_feature(graphs, 'node')
    edges = BatchGraph._build_feature(graphs, 'edge')
    edge_index = BatchGraph._build_edge_index(graphs)
    graph = BatchGraph(edge_index, nodes, None, node_graph_id,
                       edges, edge_graph_id, additional_keys=additional_keys)
    for key in additional_keys:
      item = BatchGraph._build_additional_data(graphs, key)
      graph[key] = item
    return graph

  @classmethod
  def _build_node_graph_id(cls, graphs):
    node_graph_id_list = []
    offset = 0
    for graph in graphs:
      node_graph_id_list.append(offset)
      offset += graph.num_nodes
    return np.array(node_graph_id_list, dtype=np.int64)

  @classmethod
  def _build_edge_graph_id(cls, graphs):
    edge_graph_id_list = []
    offset = 0
    for graph in graphs:
      edge_graph_id_list.append(offset)
      offset += graph.num_edges
    return np.array(edge_graph_id_list, dtype=np.int64)

  @classmethod
  def _build_feature(cls, graphs, type='node'):
    def list_append(list, item):
      if item is not None:
        list.append(item)
      return list

    def np_concat(list):
      if list:
        return np.concatenate(list, axis=0)
      return None

    #TODO(baole): support other labels and weights.
    ids_list = []
    int_attrs_list = []
    float_attrs_list = []
    string_attrs_list = []
    for graph in graphs:
      if type == 'node':
        item = graph.nodes
      else:
        return None
      # flatten format.
      ids_list = list_append(ids_list, item.ids)
      int_attrs_list = list_append(int_attrs_list, item.int_attrs)
      float_attrs_list = list_append(float_attrs_list, item.float_attrs)
      string_attrs_list = list_append(string_attrs_list, item.string_attrs)
    ids = np_concat(ids_list)
    ints = np_concat(int_attrs_list)
    floats = np_concat(float_attrs_list)
    strings = np_concat(string_attrs_list)
    return Vertex(ids, ints, floats, strings)

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
  def from_tensors(cls, tensors, node_schema, additional_keys=[], **kwargs):
    """Build SubGraph from flatten tensors.
    Args:
      tensors: A tuple of tensors corresponding to BatchGraph flatten format.
      node_schema: A (name, Decoder) tuple used to describe the nodes' feature.
      additional_keys: Keys(a list) of the additional data.
    Returns:
      BatchGraph in tensor format.
    """
    cursor = [0]
    def next(tensors):
      t = tensors[cursor[0]]
      cursor[0] += 1
      return t

    def build_node_from_tensors(feature_schema, tensors):
      """Constructs Vertices in Tensor format.
      Args:
        feature_schema: A (name, Decoder) tuple used to parse the feature.
      Returns:
        Vertex.
      """
      ids = next(tensors)
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
      feature_tensor = Vertex(ids,
                              ints=int_attrs,
                              floats=float_attrs,
                              strings=string_attrs)
      return feature_tensor

    # src
    edge_index = next(tensors)
    if node_schema is not None:
      node_tensor = build_node_from_tensors(node_schema, tensors)
      node_graph_id = next(tensors)
    else:
      node_tensor = None
      node_graph_id = None

    #TODO(baole.abl): support Edge.
    edge_tensor = None
    edge_graph_id = None

    graph = BatchGraph(edge_index, node_tensor, node_schema, node_graph_id,
      additional_keys=additional_keys)
    for key in additional_keys:
      item = next(tensors)
      graph[key] = item
    return graph