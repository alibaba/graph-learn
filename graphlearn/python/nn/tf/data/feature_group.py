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

import tensorflow as tf
from graphlearn.python.data.feature_spec import *
from graphlearn.python.nn.tf.data.feature_column import *
from graphlearn.python.nn.tf.module import Module

class FeatureGroup(Module):
  """ To merge a list of features together and output the results after fusion.

  Usually, we just need concatenate all the features. But in some cases, we need
  divide the features into groups according to their importance, and then do
  different transformations for different groups. FeatureGroup will simplify the
  procedure.

  A vertex or an edge may be made up of several feature groups.
  """

  def __init__(self, feature_column_list):
    self._fc_list = feature_column_list
    self._n = len(self._fc_list)

  def __nonzero__(self):
    return self._n > 0

  def __len__(self):
    return self._n

  def __getitem__(self, i):
    return self._fc_list[i]

  def forward(self, x_list):
    return self.dense(x_list)

  def dense(self, x_list):
    outputs = []
    if isinstance(x_list, tf.Tensor):
      num = x_list.shape[0]
    else:
      num = len(x_list)

    if self._n != num:
      raise ValueError("{} feature columns, but got {} inputs."
                       .format(self._n, num))
    for i in range(num):
      output = self._fc_list[i].dense(x_list[i])
      if isinstance(self._fc_list[i], NumericColumn):
        output = tf.expand_dims(output, axis=-1)
      outputs.append(output)
    return tf.concat(outputs, -1)


class FeatureHandler(Module):
  """ Mapping feature_spec to FeatureGroup and do reasonable optimization.

  Usage:
    handler = FeatureHandler("hop1", feature_spec)
    outputs = handler(vertex)

  Args:
    name: A unique string.
    feature_sepc: FeatureSpec object.
    fuse_embedding: Some attr in int_attrs will be embedded in a vector. If the
      dimensions of their embeddings are the same, we can fuse them together to
      improve computing efficiency.
  """

  class IndexBucket(object):
    def __init__(self):
      self.index_list = []
      self.bucket_list = []

    def append(self, index, bucket_size):
      self.index_list.append(index)
      self.bucket_list.append(bucket_size)

  def __init__(self, name, feature_spec,
               fuse_embedding=True):
    self._fspec = feature_spec
    self._fuse_embedding = fuse_embedding


    self._int_mapping = []
    self._fused_int_mapping = {}

    with tf.variable_scope("features_" + name, reuse=tf.AUTO_REUSE):
      self._float_fg = self._add_float_columns()
    with tf.variable_scope("features_" + name, reuse=tf.AUTO_REUSE):
      self._int_fg, self._fused_int_fg = self._add_int_columns()
    with tf.variable_scope("features_" + name, reuse=tf.AUTO_REUSE):
      self._string_fg = self._add_string_columns()

  def _add_float_columns(self):
    fc_list = []
    for i, spec in enumerate(self._fspec.float_specs):
      fc = NumericColumn("dense_" + str(i), spec.weight)
      fc_list.append(fc)
    return FeatureGroup(fc_list)

  def _add_int_columns(self):
    fc_list = []
    for i, spec in enumerate(self._fspec.int_specs):
      if isinstance(spec, DynamicSparseSpec):
        self._add_dynamic_embedding_column(i, spec, fc_list)
      elif isinstance(spec, SparseSpec):
        if spec.need_hash:
          self._add_embedding_column(i, spec, fc_list)
        elif self._fuse_embedding:
          self._classify_by_dimension(i, spec)
        else:
          self._add_embedding_column(i, spec, fc_list)
      else:
        # Treat integer like a float
        fc_list.append(NumericColumn("sparse_as_dense_" + str(i), spec.weight))
        self._int_mapping.append(i)

    fused_fc_list = []
    for dim, ib in self._fused_int_mapping.items():
      fc = FusedEmbeddingColumn("fused_emb_" + str(dim), ib.bucket_list, dim)
      fused_fc_list.append(fc)

    return FeatureGroup(fc_list), FeatureGroup(fused_fc_list)

  def _add_embedding_column(self, i, spec, fc_list):
    fc = EmbeddingColumn("emb_" + str(i),
        spec.bucket_size, spec.dimension, spec.need_hash)
    self._int_mapping.append(i)
    fc_list.append(fc)

  def _add_dynamic_embedding_column(self, i, spec, fc_list):
    fc = DynamicEmbeddingColumn("dynamic_int_emb_" + str(i),
        spec.dimension, is_string=False)
    self._int_mapping.append(i)
    fc_list.append(fc)

  def _classify_by_dimension(self, i, spec):
    """ Fuse the embedding columns of the same dimension together. Here we use
    self._fused_int_mapping, which is indexed by dimension and maintains all the
    feature indices and buckets.
    """
    if not spec.dimension:
      return
    elif spec.dimension in self._fused_int_mapping:
      self._fused_int_mapping[spec.dimension].append(i, spec.bucket_size)
    else:
      ib = self.IndexBucket()
      ib.append(i, spec.bucket_size)
      self._fused_int_mapping[spec.dimension] = ib

  def _add_string_columns(self):
    fc_list = []
    for i, spec in enumerate(self._fspec.string_specs):
      if isinstance(spec, DynamicSparseSpec):
        fc = DynamicEmbeddingColumn("dynamic_str_emb_" + str(i),
            spec.dimension, is_string=True)
        fc_list.append(fc)
      elif isinstance(spec, MultivalSpec):
        fc = SparseEmbeddingColumn("sparse_emb_" + str(i),
            spec.bucket_size, spec.dimension, spec.delimiter)
        fc_list.append(fc)
      elif isinstance(spec, DynamicMultivalSpec):
        fc = DynamicSparseEmbeddingColumn("dynamic_sparse_emb_" + str(i),
            spec.dimension, spec.delimiter)
        fc_list.append(fc)
    return FeatureGroup(fc_list)

  def forward(self, entity):
    outputs = []
    if self._float_fg:
      float_outputs = self._float_fg.dense(entity.float_attrs)
      outputs.append(float_outputs)

    if self._int_fg:
      ints = [entity.int_attrs[i] for i in self._int_mapping]
      int_outputs = self._int_fg.dense(ints)
      outputs.append(int_outputs)

    if self._fused_int_fg:
      fused_inputs = []
      for dim, ib in self._fused_int_mapping.items():
        fused_inputs.append([entity.int_attrs[i] for i in ib.index_list])
      fused_int_outputs = self._fused_int_fg.dense(fused_inputs)
      outputs.append(fused_int_outputs)

    if self._string_fg:
      string_outputs = self._string_fg.dense(entity.string_attrs)
      outputs.append(string_outputs)

    return tf.concat(outputs, -1)