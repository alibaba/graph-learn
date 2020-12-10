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

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

from graphlearn.python.errors import OutOfRangeError
from graphlearn.python.model.ego_graph import EgoGraph
from graphlearn.python.model.tf.ego_tensor import EgoTensor


class EgoFlow(object):
  """ EgoFlow iterates over sampled EgoGraphs using sample_seed,
  positive_sample, negative_sample and receptive_fn, and it converts
  the EgoGraphs to their TensorFlow Tensor format using EgoSpec.

  Args:
    sample_seed: the function that generate Nodes or Edges used as seed.
    positive_sample: the function used when performing positive sample.
    receptive_fn: the function used when performing neighbor sample.
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
      value = next(self._sample_generator())
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
