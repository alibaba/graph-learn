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
""" Contains the egograph flow classes which manage sampled EgoGraphs
and convert EgoGraphs to DL backend tensor format EgoTensors.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from enum import Enum
import tensorflow as tf

from graphlearn.python.gsl.dag_node import DagNode
from graphlearn.python.gsl.dag_dataset import Dataset
from graphlearn.python.errors import OutOfRangeError
from graphlearn.python.model.ego_graph import EgoGraph
from graphlearn.python.model.tf.ego_tensor import EgoTensor
from graphlearn.python.data.values import Layer

class Ego(object):
  def __init__(self, src, hops=None):
    self._src = src
    self._hops = hops

  @property
  def src(self):
    return self._src

  @property
  def hops(self):
    return self._hops

class FunctionalEgo(Ego):
  def __init__(self, func, *args):
    super(FunctionalEgo, self).__init__(None, None)
    assert callable(func), "FunctionalEgo only accpects function and it's args."
    self._func = func
    self._args = args

  def __call__(self, x):
    res = self._func(x, *self._args)
    assert len(res) == 2, \
      "Function in FunctionalEgo should return (src values, hops values)."
    self._src, self._hops = res
    return self

class EgoKeys(Enum):
  POS_SRC = 0
  NGE_SRC = 1
  POS_DST = 2
  NEG_DST = 3
  POS_EDGE = 4
  NEG_EDGE = 5

class EgoFlow(object):
  """ EgoFlow iterates over sampled EgoGraphs using sample_seed,
  positive_sample, negative_sample and receptive_fn, and it converts
  the EgoGraphs to their TensorFlow Tensor format using EgoSpec.

  Args:
    sample_seed: the function that generate Nodes or Edges used as seed.
    positive_sample: the function used when performing positive sample.
    receptive_fn: teh function used when performing neighbor sample.
    src_ego_spec: Ego spec for the source node.
    dst_ego_spec: Ego spec for the destination node.
    edge_ego_spec: Ego spec for the edge.
    negative_sample: the function used for negative sampling.
    full_graph_mode: set True if sample full graph in one batch.
  """

  def __init__(self,
               sample_seed,
               positive_sample,
               receptive_fn,
               src_ego_spec,
               dst_ego_spec=None,
               edge_ego_spec=None,
               negative_sample=None,
               full_graph_mode=False):
    """init EgoGraphFlow using sample functions and related EgoSpecs.
    """

    self._sample_seed = sample_seed
    self._positive_sample = positive_sample
    self._negative_sample = negative_sample
    self._receptive_fn = receptive_fn

    self._src_ego_spec = src_ego_spec
    self._dst_ego_spec = dst_ego_spec
    self._edge_ego_spec = edge_ego_spec

    self._full_graph_mode = full_graph_mode

    self._iterator = None
    self._pos_src_ego_tensor, \
    self._pos_dst_ego_tensor, \
    self._neg_src_ego_tensor, \
    self._neg_dst_ego_tensor, \
    self._pos_edge_ego_tensor, \
    self._neg_edge_ego_tensor = self._build()

  @property
  def iterator(self):
    return self._iterator

  @property
  def pos_src_ego_tensor(self):
    return self._pos_src_ego_tensor

  @property
  def pos_dst_ego_tensor(self):
    return self._pos_dst_ego_tensor

  @property
  def neg_src_ego_tensor(self):
    return self._neg_src_ego_tensor

  @property
  def neg_dst_ego_tensor(self):
    return self._neg_dst_ego_tensor

  @property
  def pos_edge_ego_tensor(self):
    return self._pos_edge_ego_tensor

  @property
  def neg_edge_ego_tensor(self):
    return self._neg_edge_ego_tensor

  def _feat_types_and_shapes(self, feat_spec, sparse=False):
    """Get types and shapes of FeatSpec.
    Args:
      feat_spec: A FeatureSpec object used to parse the feature.
      sparse: Bool, set to true if the feature is in the sparse format.
    Returns:
      two list of TF types and shapes.
    """
    # ids
    output_types = [tf.int64]
    output_shapes = [tf.TensorShape([None])]
    # sparse
    if sparse:
      # offsets, dense_shape, indices
      output_types.extend([tf.int64, tf.int64, tf.int64])
      output_shapes.extend([tf.TensorShape([None]),
                            tf.TensorShape([2]),
                            tf.TensorShape([None, 2])])
    # labels
    if feat_spec.labeled:
      output_types.extend([tf.int32])
      output_shapes.extend([tf.TensorShape([None])])
    # weights
    if feat_spec.weighted:
      output_types.extend([tf.float32])
      output_shapes.extend([tf.TensorShape([None])])
    # attributes
    if feat_spec.cont_attrs_num > 0:
      output_types.extend([tf.float32])
      output_shapes.extend([tf.TensorShape([None, feat_spec.cont_attrs_num])])
    if feat_spec.cate_attrs_num > 0:
      output_types.extend([tf.string])
      output_shapes.extend([tf.TensorShape([None, feat_spec.cate_attrs_num])])

    return output_types, output_shapes

  def _ego_types_and_shapes(self, ego_spec):
    """Get types and shapes of EgoSpec.

    Args:
      ego_spec: Ego spec.
    Returns:
      Two tuple of types and shapes
    """
    if ego_spec is None:
      return tuple(), tuple()

    # src(root), Nodes or Edges
    output_types, output_shapes = self._feat_types_and_shapes(ego_spec.src_spec)
    # neighbors
    if ego_spec.hops_spec is None:
      return tuple(output_types), tuple(output_shapes)

    for i in range(len(ego_spec.hops_spec)):
      # Nodes
      if ego_spec.hops_spec[i].node_spec is not None:
        nbr_ego_types, nbr_ego_shapes = self._feat_types_and_shapes(
            ego_spec.hops_spec[i].node_spec,
            sparse=ego_spec.hops_spec[i].sparse)
        output_types.extend(nbr_ego_types)
        output_shapes.extend(nbr_ego_shapes)
      # Edges
      if ego_spec.hops_spec[i].edge_spec is not None:
        nbr_ego_types, nbr_ego_shapes = self._feat_types_and_shapes(
            ego_spec.hops_spec[i].edge_spec,
            sparse=ego_spec.hops_spec[i].sparse)
        output_types.extend(nbr_ego_types)
        output_shapes.extend(nbr_ego_shapes)

    return tuple(output_types), tuple(output_shapes)

  def _sample_generator(self):
    """Sample using sample functions and return a wrapped generator

    Returns:
      Tuple of egoGraphs
    """
    while True:
      try:
        batch_seed = self._sample_seed().next()
        pos_edge = self._positive_sample(batch_seed)
        # pos src
        pos_src_recept = self._receptive_fn(pos_edge.src_nodes)

        pos_dst_recept = None
        neg_src_recept = None
        neg_dst_recept = None
        pos_edge_recept = None
        neg_edge_recept = None
        if self._edge_ego_spec is not None:
          # pose edge
          pos_edge_recept = EgoGraph(pos_edge, None)
        if self._dst_ego_spec is not None:
          # pos dst
          pos_dst_recept = self._receptive_fn(pos_edge.dst_nodes)
          if self._negative_sample is not None: # need neg
            if self._edge_ego_spec is not None:
              # negative_sample returns Edges
              neg_edge = self._negative_sample(pos_edge)
              # neg src, neg dst, neg edge.
              neg_src_recept = self._receptive_fn(neg_edge.src_nodes)
              neg_dst_recept = self._receptive_fn(neg_edge.dst_nodes)
              neg_edge_recept = EgoGraph(neg_edge, None)
            else:
              # negative_sample returns Nodes
              neg_node = self._negative_sample(pos_edge)
              neg_dst_recept = self._receptive_fn(neg_node)

        egographs_tuple = tuple()
        pos_src_flatten_egograph = \
          pos_src_recept.flatten(spec = self._src_ego_spec)
        egographs_tuple += tuple(pos_src_flatten_egograph)

        if pos_dst_recept is not None:
          pos_dst_flatten_egograph = \
            pos_dst_recept.flatten(spec = self._dst_ego_spec)
          egographs_tuple += tuple(pos_dst_flatten_egograph)

        if neg_src_recept is not None:
          neg_src_flatten_egograph = \
            neg_src_recept.flatten(spec = self._src_ego_spec)
          egographs_tuple += tuple(neg_src_flatten_egograph)

        if neg_dst_recept is not None:
          neg_dst_flatten_egograph = \
            neg_dst_recept.flatten(spec = self._dst_ego_spec)
          egographs_tuple += tuple(neg_dst_flatten_egograph)

        if pos_edge_recept is not None:
          pos_edge_flatten_egograph = \
            pos_edge_recept.flatten(spec = self._edge_ego_spec)
          egographs_tuple += tuple(pos_edge_flatten_egograph)

        if neg_edge_recept is not None:
          neg_edge_flatten_egograph = \
            neg_edge_recept.flatten(spec = self._edge_ego_spec)
          egographs_tuple += tuple(neg_edge_flatten_egograph)

        yield tuple(egographs_tuple)

      except OutOfRangeError:
        break

  def _build(self):
    """ Converts EgoGraphs to EgoTensors.
    Wraps an iterator to control EgoGraphs' flow and return EgoTensors.

    """
    output_types = tuple()
    output_shapes = tuple()

    # get types and shapes of EgoGraphs.
    src_ego_types, src_ego_shapes = \
        self._ego_types_and_shapes(self._src_ego_spec)
    self._src_ego_size = len(src_ego_types)

    dst_ego_types, dst_ego_shapes = \
        self._ego_types_and_shapes(self._dst_ego_spec)
    self._dst_ego_size = len(dst_ego_types)

    edge_ego_types, edge_ego_shapes = \
      self._ego_types_and_shapes(self._edge_ego_spec)
    self._edge_ego_size = len(edge_ego_types)

    # pos src
    output_types += src_ego_types
    output_shapes += src_ego_shapes
    if self._dst_ego_spec is not None:
      # pos dst
      output_types += dst_ego_types
      output_shapes += dst_ego_shapes
      if self._negative_sample is not None:
        if self._edge_ego_spec is not None:
          # negative_sample return edges.
          # neg src
          output_types += src_ego_types
          output_shapes += src_ego_shapes
        # neg dst
        output_types += dst_ego_types
        output_shapes += dst_ego_shapes
    # pos edge
    if self._edge_ego_spec is not None:
      output_types += edge_ego_types
      output_shapes += edge_ego_shapes
      if self._negative_sample is not None:
        # neg edge
        output_types += edge_ego_types
        output_shapes += edge_ego_shapes

    # wrap dataset.
    if self._full_graph_mode: # constant tensors
      value = self._sample_generator().next()
      value = tuple([tf.convert_to_tensor(i) for i in value])
    else:
      dataset = tf.data.Dataset.from_generator(self._sample_generator,
                                               output_types,
                                               output_shapes)
      self._iterator = dataset.make_initializable_iterator()
      value = self._iterator.get_next()

    return self._construct_ego_tensors(value)

  def _construct_ego_tensors(self, value):
    """Constructs EgoTensors using flatten tensor format of EgoGraphs.

    Args:
      value:
        An iterator to control EgoGraphs' flow
    Returns:
      EgoTensors.
    """
    offset_begin = 0
    offset_end = self._src_ego_size
    # pos src
    pos_src_ego_tensor = EgoTensor(value[offset_begin:offset_end],
                                   self._src_ego_spec)
    pos_dst_ego_tensor = None
    neg_dst_ego_tensor = None
    neg_src_ego_tensor = None
    pos_edge_ego_tensor = None
    neg_edge_ego_tensor = None
    # pos dst
    if self._dst_ego_spec is not None:
      offset_begin = offset_end
      offset_end += self._dst_ego_size
      pos_dst_ego_tensor = EgoTensor(value[offset_begin:offset_end],
                                     self._dst_ego_spec)
    # neg src
    if self._negative_sample is not None and self._edge_ego_spec is not None:
      offset_begin = offset_end
      offset_end += self._src_ego_size
      neg_src_ego_tensor = EgoTensor(value[offset_begin:offset_end],
                                     self._src_ego_spec)
    # neg dst
    if self._negative_sample is not None and self._dst_ego_spec is not None:
      offset_begin = offset_end
      offset_end += self._dst_ego_size
      neg_dst_ego_tensor = EgoTensor(value[offset_begin:offset_end],
                                     self._dst_ego_spec)
    # pos edge
    if self._edge_ego_spec is not None:
      offset_begin = offset_end
      offset_end += self._edge_ego_size
      pos_edge_ego_tensor = EgoTensor(value[offset_begin:offset_end],
                                      self._edge_ego_spec)
    # neg edge
    if self._negative_sample is not None and self._edge_ego_spec is not None:
      offset_begin = offset_end
      offset_end += self._edge_ego_size
      neg_edge_ego_tensor = EgoTensor(value[offset_begin:offset_end],
                                      self._edge_ego_spec)

    return pos_src_ego_tensor, pos_dst_ego_tensor, \
           neg_src_ego_tensor, neg_dst_ego_tensor, \
           pos_edge_ego_tensor, neg_edge_ego_tensor

class GSLEgoFlow(object):
  """ GSLEgoFlow iterates EgoGraphs in egos which constructs from results
  of sample query.

  Args:
    sample_query: a GSL query.
    egos: a dict, key is EgoKeys, value is Ego.
    ego_specs: a dict, key is EgoKeys, value is Ego spec.
  """
  def __init__(self,
               sample_query,
               egos,
               ego_specs,
               full_graph_mode=False):
    """Init EgoGraphFlow using sample functions and related EgoSpecs.
    """
    assert isinstance(sample_query, DagNode), \
      "sample_query must be a GSL query node"
    assert isinstance(egos, dict), \
      "egos must be a dict whose key is EgoKeys, value is Ego."
    self._sample_query = sample_query

    self._egos = OrderedDict(
      sorted(egos.items(),
      key=lambda t: int(getattr(EgoKeys, t[0].name).value)))
    self._ego_specs = ego_specs

    self._full_graph_mode = full_graph_mode

    self._iterator = None
    self.ds = None
    self._ego_tensors = self._build()

  @property
  def flow(self):
    return self._flow

  @property
  def iterator(self):
    return self._iterator

  def get_tensor(self, ego_key):
    assert isinstance(ego_key, EgoKeys)
    return self._ego_tensors[ego_key]

  def _feat_types_and_shapes(self, feat_spec, sparse=False):
    """Get types and shapes of FeatSpec.
    Args:
      feat_spec: A FeatureSpec object used to parse the feature.
      sparse: Bool, set to true if the feature is in the sparse format.
    Returns:
      two list of TF types and shapes.
    """
    # ids
    output_types = [tf.int64]
    output_shapes = [tf.TensorShape([None])]
    # sparse
    if sparse:
      # offsets, dense_shape, indices
      output_types.extend([tf.int64, tf.int64, tf.int64])
      output_shapes.extend([tf.TensorShape([None]),
                            tf.TensorShape([2]),
                            tf.TensorShape([None, 2])])
    # labels
    if feat_spec.labeled:
      output_types.extend([tf.int32])
      output_shapes.extend([tf.TensorShape([None])])
    # weights
    if feat_spec.weighted:
      output_types.extend([tf.float32])
      output_shapes.extend([tf.TensorShape([None])])
    # attributes
    if feat_spec.cont_attrs_num > 0:
      output_types.extend([tf.float32])
      output_shapes.extend([tf.TensorShape([None, feat_spec.cont_attrs_num])])
    if feat_spec.cate_attrs_num > 0:
      output_types.extend([tf.int32])
      output_shapes.extend([tf.TensorShape([None, feat_spec.cate_attrs_num])])

    return output_types, output_shapes

  def _ego_types_and_shapes(self, ego_spec):
    """Get types and shapes of EgoSpec.

    Args:
      ego_spec: Ego spec.
    Returns:
      Two tuple of types and shapes
    """
    if ego_spec is None:
      return tuple(), tuple()

    # src(root), Nodes or Edges
    output_types, output_shapes = self._feat_types_and_shapes(ego_spec.src_spec)
    # neighbors
    if ego_spec.hops_spec is None:
      return tuple(output_types), tuple(output_shapes)

    for i in range(len(ego_spec.hops_spec)):
      # Nodes
      if ego_spec.hops_spec[i].node_spec is not None:
        nbr_ego_types, nbr_ego_shapes = self._feat_types_and_shapes(
            ego_spec.hops_spec[i].node_spec,
            sparse=ego_spec.hops_spec[i].sparse)
        output_types.extend(nbr_ego_types)
        output_shapes.extend(nbr_ego_shapes)
      # Edges
      if ego_spec.hops_spec[i].edge_spec is not None:
        nbr_ego_types, nbr_ego_shapes = self._feat_types_and_shapes(
            ego_spec.hops_spec[i].edge_spec,
            sparse=ego_spec.hops_spec[i].sparse)
        output_types.extend(nbr_ego_types)
        output_shapes.extend(nbr_ego_shapes)

    return tuple(output_types), tuple(output_shapes)

  def _construct_ego_graph(self, data):
    res = []
    for key, ego in self._egos.items():
      assert key in self._ego_specs.keys(), \
        "EgoSpec for {} not existed.".format(key.name)
      if isinstance(ego, FunctionalEgo):
        ego = ego(data)
        src_value, hops_values = ego.src, ego.hops
      elif isinstance(ego, Ego):
        src, hops = ego.src, ego.hops
        src_value = data[src]
        if hops:
          hops_values = [Layer(nodes=data[hop]) for hop in hops]
        else:
          hops_values = None
      else:
        raise ValueError("EgoFlow only accecpets egos whose values are Egos.")
      res += EgoGraph(src_value, hops_values).flatten(self._ego_specs[key])
    return tuple(res)

  def _dataset(self):
    query = self._sample_query
    return query.values(lambda x: self._construct_ego_graph(x))

  def _sample_generator(self):
    """Sample using sample functions and return a wrapped generator

    Returns:
      Tuple of egoGraphs
    """
    while True:
      try:
        yield self.ds.next()
      except OutOfRangeError:
        break

  def _build(self):
    """ Converts EgoGraphs to EgoTensors.
    Wraps an iterator to control EgoGraphs' flow and return EgoTensors.

    """
    output_types = tuple()
    output_shapes = tuple()

    ego_sizes = {}

    # get types and shapes of EgoGraphs.
    for key in self._egos.keys():
      assert key in self._ego_specs.keys(), \
        "EgoSpec for {} not existed.".format(key.name)
      spec = self._ego_specs[key]
      ego_types, ego_shapes = self._ego_types_and_shapes(spec)
      ego_sizes[key] = len(ego_types)
      output_types += ego_types
      output_shapes += ego_shapes

    # wrap dataset.
    if self._full_graph_mode: # constant tensors
      value = self._sample_generator().next()
      value = tuple([tf.convert_to_tensor(i) for i in value])
    else:
      self.ds = Dataset(self._dataset())
      dataset = tf.data.Dataset.from_generator(self._sample_generator,
                                               output_types,
                                               output_shapes)
      self._iterator = dataset.make_initializable_iterator()
      value = self._iterator.get_next()

    return self._construct_ego_tensors(value, ego_sizes)

  def _construct_ego_tensors(self, value, sizes):
    """Constructs EgoTensors using flatten tensor format of EgoGraphs.

    Args:
      value:
        An iterator to control EgoGraphs' flow
      sizes:
        A dict, key is EgoKeys, value is int, which indicates the size
        of the corresponding EgoGraph.
    Returns:
      A dict of EgoTensors, key is EgoKeys.
    """
    ego_tensors = {}
    offset_begin = 0
    offset_end = 0
    for key in self._egos.keys():
      offset_begin = offset_end
      offset_end += sizes[key]
      ego_tensors[key] = EgoTensor(value[offset_begin:offset_end], self._ego_specs[key])
    return ego_tensors
