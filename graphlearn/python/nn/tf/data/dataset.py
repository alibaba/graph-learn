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

from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python.errors import OutOfRangeError
from graphlearn.python.gsl.dag_node import TraverseEdgeDagNode
from graphlearn.python.nn.dataset import Dataset as RawDataset
from graphlearn.python.nn.subgraph import SubGraph
from graphlearn.python.nn.tf.data.batchgraph import BatchGraph
from graphlearn.python.nn.tf.data.egograph import EgoGraph
from graphlearn.python.nn.tf.data.hetero_batchgraph import HeteroBatchGraph


class Dataset(object):
  """`Dataset` object is used to convert GSL query results to Tensor format
  `Data`. It provides methods to get raw `Data` dict, `EgoGraph`, `SubGraph` 
  and `HeteroSubGraph`.

  Args:
    query: GSL query.
    window: dataset capacity.
    inducer: A `SubGraphInducer` instance to generate SubGraph/HeteroSubGraph.
  """
  def __init__(self, query, window=10, inducer=None, **kwargs):
    self._dag = query
    self.inducer = inducer
    self._additional_keys = []
    self.node_types = self._dag.node_types
    self.edge_types = self._dag.edge_types
    self.use_edges, self.use_neg = False, False
    if self.inducer is not None:
      if self.inducer.addl_types_and_shapes is not None:
        self._additional_keys = self.inducer.addl_types_and_shapes.keys()
      if self.inducer.edge_types is not None:
        self.edge_types = self.inducer.edge_types
      if self.inducer.node_types is not None:
        self.node_types = self.inducer.node_types
      self.use_edges = self.inducer.use_edges
      self.use_neg = self.inducer.use_neg
    self._rds = RawDataset(query, window)
    self._iterator = self._make_iterator()
    # Here we call tensorflow `Iterator.get_next()` to get values 
    # in each iteration of the training loop, and then we form these values to 
    # data_dict, `EgoGraph`s and `BatchGraph`s. 
    # Note: This function must be called only once in a training loop.
    self._values = self._iterator.get_next()

  @property
  def iterator(self):
    return self._iterator
  
  def get_data_dict(self):
    """get a dict of tensor format `Data` corresponding the given query.
    Keys of the dict is the aliaes in query.
    """
    return self._rds.build_data_dict(list(self._values))

  def get_egograph(self, source, neighbors=None):
    """ Origanizes the data dict as EgoGraphs and then check and return
    the specified `EgoGraph`.
    Args:
      source(str): alias of centric vertices.
      neighbors(list of str): alias of neighbors at each hop.
        Default `None`: automatically generating the positive neighbors for
        centric vertices. It requires that each hop only has one postive
        downstream in GSL.
        Given list of string: the alias of each hop in GSL. The list must
        follow the order of traverse in GSL, and each one should be the postive
        or negative downstream for the front.
    """
    data_dict = self.get_data_dict()
    def _get_feat_spec(alias):
      node = self._dag.get_node(alias)
      decoder = node.decoder
      return decoder.feature_spec

    source_node = self._dag.get_node(source)
    nbr_nodes, nbr_edges, nbr_nums = [], [], []
    if neighbors:
      # Use specified neighbors to construct EgoGraph.
      if not isinstance(neighbors, list):
        raise ValueError("`neighbors` should be a list of alias")
      pre = source_node
      for nbr in neighbors:
        dag_node = self._dag.get_node(nbr)
        if not dag_node in pre.pos_downstreams + pre.neg_downstreams:
          raise ValueError("{} is not the downstream of {}.".format(
            dag_node.get_alias(), pre.get_alias()))
        if isinstance(pre, TraverseEdgeDagNode):
          nbr_edges.append(dag_node.get_alias())
        else:
          nbr_nodes.append(dag_node.get_alias())
          nbr_nums.append(dag_node.shape[-1])
        pre = dag_node
    else:
      # Use default receptive neighbors to construct EgoGraph.
      pre = source_node
      recepts = source_node.pos_downstreams
      while recepts:
        if len(recepts) > 1:
          raise ValueError("Can't automatically find neighbors for {},"
                           " which has multiple downstreams. You should"
                           " assign specific neighbors for {}."
                           .format(pre.get_alias(), source))
        cur = recepts[0]
        if isinstance(cur, TraverseEdgeDagNode):
          nbr_edges.append(cur.get_alias())
        else: 
          nbr_nodes.append(cur.get_alias())
          nbr_nums.append(cur.shape[-1])
        recepts = cur.pos_downstreams
    return EgoGraph(data_dict[source],
                    [data_dict[nbr] for nbr in nbr_nodes],
                    [(self._dag.get_node(v).type, _get_feat_spec(v)) for v in [source] + nbr_nodes],
                    nbr_nums,
                    [data_dict[nbr] for nbr in nbr_edges],
                    [(self._dag.get_node(v).type, _get_feat_spec(v)) for v in nbr_edges])

  def get_batchgraph(self):
    """get `BatchGraph`/`HeteroBatchGraph`s. 
    """
    neg_graph = None
    graph = BatchGraph
    node_schema = [(x, self._dag.graph.get_node_decoder(x)) for x in self.node_types]
    edge_schema = [(x, self._dag.graph.get_node_decoder(x)) for x in self.edge_types]
    if len(self.node_types) > 1 or len(self.edge_types) > 1: # HeteroBatchGraph
      graph = HeteroBatchGraph
    else:
      node_schema, edge_schema = node_schema[0], edge_schema[0]
    pos_graph = graph.from_tensors(self._values[0:self.pos_size], 
                                   node_schema, edge_schema,
                                   use_edges=self.use_edges,
                                   additional_keys=self._additional_keys)
    if self.use_neg:
      neg_graph = graph.from_tensors(self._values[self.pos_size:],
                                     node_schema, edge_schema,
                                     use_edges=self.use_edges,
                                     additional_keys=self._additional_keys)
    return pos_graph, neg_graph

  def _make_iterator(self):
    # for Data dict and EgoGraph.
    if self.inducer is None:
      output_types, output_shapes = self._raw_flatten_types_and_shapes()
      generator = self._raw_flatten_generator
    # for BatchGraph/HeteroBatchGraph.
    else: 
      output_types, output_shapes = self._batchgraph_flatten_types_and_shapes()
      generator = self._batchgraph_flatten_generator
    dataset = tf.data.Dataset.from_generator(generator,
                                             tuple(output_types),
                                             tuple(output_shapes))
    return dataset.make_initializable_iterator()

  def _raw_flatten_types_and_shapes(self):
    output_types = []
    output_shapes = []
    for alias in self._rds.masks.keys():
      node = self._dag.get_node(alias)
      types, shapes = self._data_types_and_shapes(node.decoder, 
                                                  is_edge=isinstance(node, TraverseEdgeDagNode),
                                                  is_sparse=node.sparse)
      output_types.extend(types)
      output_shapes.extend(shapes)
    return output_types, output_shapes

  def _raw_flatten_generator(self):
    while True:
      try:
        yield tuple(self._rds.get_flatten_values())
      except OutOfRangeError:
        break

  def _batchgraph_flatten_types_and_shapes(self):
    output_types, output_shapes = self._batchgraph_types_and_shapes()
    self.pos_size = len(output_types)
    if self.use_neg:
      neg_types, neg_shapes = self._batchgraph_types_and_shapes()
      output_types += neg_types
      output_shapes += neg_shapes
    return output_types, output_shapes

  def _batchgraph_flatten_generator(self):
    while True:
      try:
        subgraphs, neg_subgraphs = self._rds.get_subgraphs(self.inducer)
        if isinstance(subgraphs[0], SubGraph):
          pos_batchgraph = BatchGraph.from_graphs(subgraphs, 
                                                  self._additional_keys)
          flatten_list = pos_batchgraph.flatten()
          if self.use_neg and neg_subgraphs is not None:
            neg_batchgraph = BatchGraph.from_graphs(neg_subgraphs, 
                                                    self._additional_keys)
            flatten_list.extend(neg_batchgraph.flatten())
        else: # HeteroSubGraph.
          pos_batchgraph = HeteroBatchGraph.from_graphs(subgraphs)
          flatten_list = pos_batchgraph.flatten(self.node_types, 
                                                self.edge_types,
                                                self.use_edges)
          if self.use_neg and neg_subgraphs is not None:
            neg_batchgraph = HeteroBatchGraph.from_graphs(neg_subgraphs)
            flatten_list.extend(neg_batchgraph.flatten(self.node_types, 
                                                       self.edge_types,
                                                       self.use_edges))
        yield tuple(flatten_list)
      except OutOfRangeError:
        break

  def _data_types_and_shapes(self, node_decoder, is_edge=False, is_sparse=False):
    feat_masks, id_masks, sparse_masks = self._rds.get_mask(node_decoder, is_edge, is_sparse)
    feat_types = np.array([tf.int64, tf.float32, tf.string,
                           tf.int32, tf.float32])[feat_masks]
    feat_shapes = np.array([tf.TensorShape([None, node_decoder.int_attr_num]),
                            tf.TensorShape([None, node_decoder.float_attr_num]),
                            tf.TensorShape([None, node_decoder.string_attr_num]),
                            tf.TensorShape([None]),  # labels
                            tf.TensorShape([None])])[feat_masks] # weights

    id_types = np.array([tf.int64, tf.int64])[id_masks] # ids, dst_ids
    id_shapes = np.array([tf.TensorShape([None]), tf.TensorShape([None])])[id_masks]
    # offsets, indices and dense_shape for sparse Data.
    sparse_types = np.array([tf.int64, tf.int64, tf.int64])[sparse_masks]
    sparse_shapes = np.array([tf.TensorShape([None]), 
                              tf.TensorShape([None, 2]), 
                              tf.TensorShape([None])])[sparse_masks]
    return list(feat_types) + list(id_types) + list(sparse_types), \
      list(feat_shapes) + list(id_shapes) + list(sparse_shapes)

  def _batchgraph_types_and_shapes(self):
    output_types, output_shapes = tuple(), tuple()
    # edge index
    for edge in self.edge_types:
      output_types += tuple([tf.int32])
      output_shapes += tuple([tf.TensorShape([2, None])])
    # nodes
    for node in self.node_types:
      node_types, node_shapes = \
        self._data_types_and_shapes(self._dag.graph.get_node_decoder(node))
      output_types += tuple(node_types)
      output_shapes += tuple(node_shapes)
      # graph_node_offsets
      output_types += tuple([tf.int64])
      output_shapes += tuple([tf.TensorShape([None])])
    # edges
    if self.use_edges:
      for edge in self.edge_types:
        edge_types, edge_shapes = \
          self._data_types_and_shapes(self._dag.graph.get_node_decoder(edge))
        output_types += tuple(edge_types)
        output_shapes += tuple(edge_shapes)
        # graph_edge_offsets
        output_types += tuple([tf.int64])
        output_shapes += tuple([tf.TensorShape([None])])
    # additional data.
    for key in self._additional_keys:
      output_types += tuple([self.inducer.addl_types_and_shapes[key][0]])
      output_shapes += tuple([self.inducer.addl_types_and_shapes[key][1]])
    return output_types, output_shapes