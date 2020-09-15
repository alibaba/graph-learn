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
""" Values include Nodes, Edges, Layer, Layers that returned
by samplers. Values should be extended with customized samplers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from graphlearn import pywrap_graphlearn as pywrap
from graphlearn.python.decoder import Decoder
import graphlearn.python.utils as utils
from graphlearn.python.errors import raise_exception_on_not_ok_status


class Values(object):
  """ Base value class.
  """

  def __init__(self,
               int_attrs=None,
               float_attrs=None,
               string_attrs=None,
               weights=None,
               labels=None,
               shape=None,
               graph=None):
    self._weights = weights
    self._labels = labels
    self._int_attrs = int_attrs
    self._float_attrs = float_attrs
    self._string_attrs = string_attrs
    self._shape = shape
    self._graph = graph
    self._attred = True
    if self._int_attrs is None and self._float_attrs is None and \
      self._string_attrs is None:
      self._attred = False

  def _get_decoder(self):
    return Decoder()

  @property
  def int_attrs(self):
    self._set_attrs()
    return self._int_attrs

  @property
  def float_attrs(self):
    self._set_attrs()
    return self._float_attrs

  @property
  def string_attrs(self):
    self._set_attrs()
    return self._string_attrs

  @property
  def weights(self):
    if self._weights is None and self._get_decoder().weighted:
      self._lookup_and_set_values()
    return self._weights

  @property
  def labels(self):
    if self._labels is None and self._get_decoder().labeled:
      self._lookup_and_set_values()
    return self._labels

  @property
  def shape(self):
    return self._shape

  @property
  def graph(self):
    return self._graph

  @int_attrs.setter
  def int_attrs(self, int_attrs):
    self._int_attrs = self._reshape(int_attrs, expand_shape=True)

  @float_attrs.setter
  def float_attrs(self, float_attrs):
    self._float_attrs = self._reshape(float_attrs, expand_shape=True)

  @string_attrs.setter
  def string_attrs(self, string_attrs):
    self._string_attrs = self._reshape(string_attrs, expand_shape=True)

  @weights.setter
  def weights(self, weights):
    self._weights = self._reshape(weights)

  @labels.setter
  def labels(self, labels):
    self._labels = self._reshape(labels)

  @shape.setter
  def shape(self, shape):
    self._shape = shape
    if not isinstance(shape, tuple):
      raise ValueError("shape must be a tuple, got {}.".format(type(shape)))

  @graph.setter
  def graph(self, graph):
    self._graph = graph

  def _reshape(self, value, expand_shape=False):
    """ Reshape for value when `shape` is not None.
    """
    if value is None or value.size == 0:
      return value
    if not isinstance(value, np.ndarray):
      raise ValueError("{} must be a numpy.ndarray.".format(value))
    if self._shape:
      if not isinstance(self._shape, tuple):
        raise ValueError("shape must be a tuple, got {}."
                         .format(type(self._shape)))
      if expand_shape:
        return np.reshape(value, self._shape + (-1, ))  # pylint: disable=bad-whitespace
      return np.reshape(value, self._shape)
    return value

  def _lookup_and_set_values(self):
    pass

  def _set_attrs(self):
    if self._get_decoder().attributed:
      if not self._attred:
        self._lookup_and_set_values()
        self._attred = True

  def _set_values(self, values):
    self.int_attrs = values.int_attrs
    self.float_attrs = values.float_attrs
    self.string_attrs = values.string_attrs
    self.weights = values.weights
    self.labels = values.labels


class SparseBase(object):
  """ Sparse Value, the base class of SparseNodes and SparseEdges.
  """

  def __init__(self, offsets, dense_shape):
    """ Init a SparseBase object.
    Args:
      offsets: list or 1D ndarraay, the number of values on each line.
      dense_shape: the corresponding 2D dense shape.
    """
    self._it = 0
    self._offsets = offsets
    self._dense_shape = dense_shape
    self._global_offsets = [0]
    sum_offsets = 0
    for offset in self._offsets:
      sum_offsets += offset
      self._global_offsets.append(sum_offsets)

  @property
  def offsets(self):
    return self._offsets

  @property
  def indices(self):
    indices = []
    for x in range(len(self._offsets)):
      for y in range(self._offsets[x]):
        indices.append([x, y])
    return indices

  @property
  def dense_shape(self):
    return self._dense_shape

  @offsets.setter
  def offsets(self, offsets):
    """ row offsets
    """
    self._offsets = offsets

  @dense_shape.setter
  def dense_shape(self, dense_shape):
    self._dense_shape = dense_shape

  def __iter__(self):
    return self

  def __next__(self):
    pass

  def next(self):
    return self.__next__()


class Nodes(Values):
  """ As returned object of `get_next` api of `node_sampler` and
  `negative_sampler`, as returned object of `get_nodes` of `Graph`
  or as in-memory object for constructing graph.
  """

  def __init__(self,
               ids,
               node_type,
               int_attrs=None,
               float_attrs=None,
               string_attrs=None,
               weights=None,
               labels=None,
               shape=None,
               graph=None):
    super(Nodes, self).__init__(int_attrs=int_attrs,
                                float_attrs=float_attrs,
                                string_attrs=string_attrs,
                                weights=weights,
                                labels=labels,
                                shape=shape,
                                graph=graph)

    if not isinstance(ids, np.ndarray):
      raise ValueError("ids must be an instance of numpy.ndarray, "
                       "got {}.".format(type(ids)))
    self._shape = shape if shape is not None else ids.shape
    self._ids = self._reshape(ids)
    self._type = node_type

  def _get_decoder(self):
    return self._graph.get_node_decoder(self._type)

  @property
  def ids(self):
    return self._ids

  @property
  def type(self):  # pylint: disable=redefined-builtin
    return self._type

  @property
  def shape(self):
    return self._shape

  def _lookup_and_set_values(self):
    values = self._graph.lookup_nodes(self._type, self._ids)
    self._set_values(values)

  @ids.setter
  def ids(self, ids):
    self._ids = self._reshape(ids)

  @type.setter
  def type(self, node_type):  # pylint: disable=redefined-builtin
    self._type = node_type

  def _agg(self, func, segment_ids, num_segments):
    req = pywrap.new_aggregating_request(
        self._type, utils.strategy2op(func, "Aggregator"))
    pywrap.set_aggregating_request(req, self._ids.flatten(),
                                   np.array(segment_ids, dtype=np.int32),
                                   num_segments)
    res = pywrap.new_aggregating_response()
    status = self.graph.get_client().agg_nodes(req, res)
    if status.ok():
      agged = pywrap.get_aggregating_nodes(res)
    pywrap.del_op_response(res)
    pywrap.del_op_request(req)
    raise_exception_on_not_ok_status(status)
    return agged

  def embedding_agg(self, func="sum"):
    """
    Get aggregated embedding of fixed size of neighbors of batch seed nodes.
    The shape of neighbors embeddings is
    `[batch_size, num_neighbors, float_attr_num]`, after aggregation on axis=1,
    the shape is `[batch_size, float_attr_num]`.

    Args:
      func ("sum" | "mean" | "min" | "max" | "prod"):
        the built-in aggregate functions.
    """
    if not len(self.shape) == 2:
      raise ValueError("embedding_agg is for Nodes with 2 dimension,"
                       "and the default aggregated dimension is axis=1")
    segment_ids = \
        [i for i in range(self.shape[0]) for _ in range(self.shape[1])]
    agged =  self._agg(func, segment_ids, self.shape[0])
    return np.reshape(agged,
                      (self.shape[0], self._get_decoder().float_attr_num))

class SparseNodes(Nodes, SparseBase):
  """ SparseNodes is the returned value of full neighbor sampler which
  is 2D. It can be easily transformed to Tensorflow or PyTorch Sparse
  Tensors.
  """

  def __init__(self,
               ids,
               offsets,
               dense_shape,
               node_type,
               int_attrs=None,
               float_attrs=None,
               string_attrs=None,
               weights=None,
               labels=None,
               graph=None):
    """ Sparse Nodes.
    Args:
      ids: A 1D numpy array, the ids of the nodes.
      offsets: A python list, each elem of list is an int,
        which indicates the number of nodes.
      dense_shape: The shape of the the corresponding dense Nodes.
      For example, ids=[5, 2, 1, 6, 2, 4],
      offsets=[3, 2, 1],
      dense_shape=[3, 5].
      The corresponding dense Nodes is
      [[ 5,  2,  1, -1, -1],
       [ 6,  2, -1, -1, -1],
       [ 4, -1, -1, -1, -1]]
    """
    Nodes.__init__(self, ids,
                   node_type,
                   int_attrs=None,
                   float_attrs=None,
                   string_attrs=None,
                   weights=weights,
                   labels=labels,
                   shape=None,
                   graph=graph)
    SparseBase.__init__(self, offsets, dense_shape)
    num_nodes = sum(offsets)
    if ids.shape[0] != num_nodes:
      raise ValueError("Ids must be the same length of indices")

  def __next__(self):
    if self._it < len(self._offsets):
      l = self._global_offsets[self._it]
      r = self._global_offsets[self._it + 1]
      self._it += 1
      nodes = Nodes(self._ids[l: r], self._type, graph=self._graph,
                    int_attrs=np.array([int_attr[l: r] \
                        for int_attr in self._int_attrs]) \
                        if self._int_attrs is not None else None,
                    float_attrs=np.array([float_attr[l: r] \
                        for float_attr in self._float_attrs]) \
                        if self._float_attrs is not None else None, \
                    string_attrs=np.array([string_attr[l: r] \
                        for string_attr in self._string_attrs]) \
                        if self._string_attrs is not None else None,
                    weights=self._weights[l:r] \
                        if self._weights is not None else None,
                    labels=self._labels[l:r] \
                        if self._labels is not None else None)
      return nodes
    else:
      raise StopIteration

  def embedding_agg(self, func="sum"):
    """
    Get aggregated embeddings of full neighbors of batch seed nodes.
    The shape of neighbors embeddings is `[total_num_nbrs, float_attr_num]`.
    After aggregation, the shape is `[redcued_num_nbrs, float_attr_num]`.

    Args:
      func ("sum" | "mean" | "min" | "max" | "prod"):
        the built-in aggregate functions.
    """
    float_attr_num = self._get_decoder().float_attr_num
    batch_size = len(self.offsets)
    segment_ids = \
        [i for i in range(batch_size) for _ in range(self.offsets[i])]
    agged = self._agg(func, segment_ids, batch_size)
    return np.reshape(agged, (batch_size, float_attr_num))


class Edges(Values):
  """ As returned object of `get_next` api of `edge_sampler` ,
  as returned object of `get_edges` of `Graph` or as in-memory object
  for constructing graph.
  """

  def __init__(self,
               src_ids=None,
               src_type=None,
               dst_ids=None,
               dst_type=None,
               edge_type=None,
               edge_ids=None,
               src_nodes=None,
               dst_nodes=None,
               int_attrs=None,
               float_attrs=None,
               string_attrs=None,
               weights=None,
               labels=None,
               shape=None,
               graph=None):
    super(Edges, self).__init__(int_attrs=None,
                                float_attrs=None,
                                string_attrs=None,
                                weights=weights,
                                labels=labels,
                                shape=shape,
                                graph=graph)
    self._shape = shape
    if not self._shape:
      if src_ids is not None:
        if not isinstance(src_ids, np.ndarray):
          raise ValueError("src_ids must be an instance of numpy.ndarray, "
                           "got {}.".format(type(src_ids)))
        self._shape = src_ids.shape
      if edge_ids is not None:
        if not isinstance(dst_ids, np.ndarray):
          raise ValueError("dst_ids must be an instance of numpy.ndarray, "
                           "got {}.".format(type(dst_ids)))
        self._shape = edge_ids.shape
    self._src_ids = self._reshape(src_ids)
    self._src_type = src_type
    self._dst_ids = self._reshape(dst_ids)
    self._dst_type = dst_type
    self._edge_type = edge_type
    self._edge_ids = self._reshape(edge_ids)
    self._src_nodes = src_nodes
    self._dst_nodes = dst_nodes

    if self._src_ids is not None and self._src_nodes is None:
      self._src_nodes = Nodes(src_ids, src_type, shape=shape, graph=graph)

    if self._dst_ids is not None and self._dst_nodes is None:
      self._dst_nodes = Nodes(dst_ids, dst_type, shape=shape, graph=graph)

    if self._src_ids is not None and self._dst_ids is not None:
      if self._src_ids.shape != self._dst_ids.shape:
        raise ValueError("src_ids and dst_ids must be same shape.")

  def _get_decoder(self):
    return self._graph.get_edge_decoder(self._edge_type)

  @property
  def src_nodes(self):
    return self._src_nodes

  @property
  def dst_nodes(self):
    return self._dst_nodes

  @property
  def edge_ids(self):
    return self._edge_ids

  @property
  def src_ids(self):
    return self._src_ids

  @property
  def dst_ids(self):
    return self._dst_ids

  @property
  def src_type(self):
    return self._src_type

  @property
  def dst_type(self):
    return self._dst_type

  @property
  def edge_type(self):
    return self._edge_type

  @property
  def type(self):  # pylint: disable=redefined-builtin
    return self._src_type, self._dst_type, self._edge_type

  @property
  def shape(self):
    return self._shape

  def _lookup_and_set_values(self):
    values = self._graph.lookup_edges(self._edge_type,
                                      self._src_ids,
                                      self._edge_ids)
    self._set_values(values)

  @edge_ids.setter
  def edge_ids(self, edge_ids):
    self._edge_ids = self._reshape(edge_ids)

  @src_ids.setter
  def src_ids(self, src_ids):
    self._src_ids = self._reshape(src_ids)

  @dst_ids.setter
  def dst_ids(self, dst_ids):
    self._dst_ids = self._reshape(dst_ids)

  @type.setter
  def type(self, type):  # pylint: disable=redefined-builtin
    if not isinstance(type, tuple) or len(type) != 3:
      raise ValueError("property type must be a tuple of "
                       "(src_type, dst_type, edge_type).")
    self._src_type, self._dst_type, self._edge_type = type

  @src_nodes.setter
  def src_nodes(self, src_nodes):
    if not isinstance(src_nodes, Nodes):
      raise ValueError("property src_nodes must be a Nodes object.")
    self._src_nodes = src_nodes

  @dst_nodes.setter
  def dst_nodes(self, dst_nodes):
    if not isinstance(dst_nodes, Nodes):
      raise ValueError("property dst_nodes must be a Nodes object.")
    self._dst_nodes = dst_nodes


class SparseEdges(Edges, SparseBase):
  """ SparseEdges is the return value of full neighbor sampler.
  It can be easily transformed to Tensorflow or PyTorch Sparse Tensors.
  """

  def __init__(self,
               src_ids=None,
               src_type=None,
               dst_ids=None,
               dst_type=None,
               edge_type=None,
               offsets=None,
               dense_shape=None,
               edge_ids=None,
               src_nodes=None,
               dst_nodes=None,
               int_attrs=None,
               float_attrs=None,
               string_attrs=None,
               weights=None,
               labels=None,
               graph=None):
    """ Sparse Edges.
    """
    Edges.__init__(self, src_ids=src_ids,
                   src_type=src_type,
                   dst_ids=dst_ids,
                   dst_type=dst_type,
                   edge_type=edge_type,
                   edge_ids=edge_ids,
                   src_nodes=src_nodes,
                   dst_nodes=dst_nodes,
                   int_attrs=None,
                   float_attrs=None,
                   string_attrs=None,
                   weights=weights,
                   labels=labels,
                   shape=None,
                   graph=graph)
    SparseBase.__init__(self, offsets, dense_shape)
    if not src_nodes:
      num_edges = sum(offsets)
      if src_ids is not None and src_ids.shape[0] != num_edges:
        raise ValueError("Ids must be the same length of indices")
      self._src_nodes = SparseNodes(src_ids, offsets, dense_shape,
                                    src_type, graph=graph)
      self._dst_nodes = SparseNodes(dst_ids, offsets, dense_shape,
                                    dst_type, graph=graph)
    else:
      self._dense_shape = dst_nodes.dense_shape
      self._offsets = dst_nodes.offsets

  def __next__(self):
    if self._it < len(self._offsets):
      l = self._global_offsets[self._it]
      r = self._global_offsets[self._it + 1]
      self._it += 1
      edges = Edges(self._src_ids[l: r] \
                      if self._src_ids is not None else None,
                    self._src_type,
                    self._dst_ids[l: r] \
                      if self._dst_ids is not None else None,
                    self._dst_type,
                    self._edge_type,
                    self._edge_ids[l: r] \
                      if self._edge_ids is not None else None,
                    next(self._src_nodes),
                    next(self._dst_nodes),
                    weights=self._weights[l:r] \
                      if self._weights is not None else None,
                    labels=self._labels[l:r] \
                      if self._labels is not None else None,
                    graph=self._graph)
      edges.int_attrs = np.array(
          [int_attr[l: r] for int_attr in self._int_attrs]) \
          if self._int_attrs is not None else None
      edges.float_attrs = np.array(
          [float_attr[l: r] for float_attr in self._float_attrs]) \
          if self._float_attrs is not None else None
      edges.string_attrs = np.array(
          [string_attr[l: r] for string_attr in self._string_attrs]) \
          if self._string_attrs is not None else None
      return edges
    else:
      raise StopIteration


class Layers(object):
  """ As returned object of `get_next` api of `meta_path_sampler`.
  """

  def __init__(self, layers=None):
    self.layers = layers if layers else []

  def layer(self, layer_id):
    """ Get one `Layer`.
    """
    layer_id -= 1
    if isinstance(self.layers, list) and layer_id < len(self.layers):
      return self.layers[layer_id]
    else:
      raise ValueError("layer id beyond the layers length.")

  def layer_size(self, layer_id):
    """ Get size of the given `Layer`.
    """
    layer_id -= 1
    if isinstance(self.layers, list) and layer_id < len(self.layers):
      return self.layers[layer_id].shape
    else:
      raise ValueError("layer id beyond the layers length.")

  def layer_nodes(self, layer_id):
    """ Get `Nodes` of the given `Layer`.
    """
    layer_id -= 1
    if isinstance(self.layers, list) and layer_id < len(self.layers):
      return self.layers[layer_id].nodes
    else:
      raise ValueError("layer id beyond the layers length.")

  def layer_edges(self, layer_id):
    """ Get `Edges` of the given `Layer`.
    """
    layer_id -= 1
    if isinstance(self.layers, list) and layer_id < len(self.layers):
      return self.layers[layer_id].edges
    else:
      raise ValueError("layer id beyond the layers length.")

  def set_layer_nodes(self, layer_id, nodes):
    """ Set `Nodes` of the given `Layer`.
    """
    layer_id -= 1
    if isinstance(self.layers, list) and layer_id < len(self.layers):
      if isinstance(self.layers[layer_id], Layer):
        self.layers[layer_id].set_nodes(nodes)
      else:
        raise ValueError("layer {} is not a SingleLayer".format(layer_id))
    else:
      raise ValueError("layer id beyond the layers length.")

  def set_layer_edges(self, layer_id, edges):
    """ Set `Edges` of the given `Layer`.
    """
    layer_id -= 1
    if isinstance(self.layers, list) and layer_id < len(self.layers):
      if isinstance(self.layers[layer_id], Layer):
        self.layers[layer_id].set_edges(edges)
      else:
        raise ValueError("layer {} is not a SingleLayer".format(layer_id))
    else:
      raise ValueError("layer id beyond the layers length.")

  def append_layer(self, layer):
    """ Append a `Layer` to layers
    """
    self.layers.append(layer)


class Layer(object):
  """ Layer is 1 hop neighbor nodes and the between edges.
  """

  def __init__(self, nodes, edges=None, shape=None):
    """ A `Layer` maintain one hop of `Nodes` and `Edges`."""
    self._nodes = nodes
    self._edges = edges
    self._shape = shape if shape else nodes.shape

  @property
  def nodes(self):
    return self._nodes

  @property
  def edges(self):
    return self._edges

  @property
  def shape(self):
    return self._shape

  @nodes.setter
  def nodes(self, nodes):
    self._nodes = nodes

  @edges.setter
  def edges(self, edges):
    self._edges = edges

  @shape.setter
  def shape(self, shape):
    self._shape = shape
