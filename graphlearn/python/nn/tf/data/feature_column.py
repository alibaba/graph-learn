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

import json
import os

import numpy as np
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

from graphlearn.python.nn.tf.config import conf
from graphlearn.python.nn.tf.module import Module


class FeatureColumn(Module):
  """ Transforms raw features to dense tensors. For continuous features, just 
  return the original values, for categorical features, embeds them to dense
  vectors.

  For example, each 'user' vertex in the graph contains 6 attributes splited by
  ':', which looks like '28:0:0.2:Hangzhou:1,5,12:1000008'. To handle such a
  vertex, 6 `FeatureColumn` objects are needed, each of which will return a
  dense value. And then we will concat all the dense values together to get the
  representation of this vertex.

  Each feature can be configured differently. The first two features, 28 and 0,
  are categorical, and both of them will be encoded into continuous spaces with
  dimension 12. To improve the efficiency, we can fuse the two spaces together
  to minimize the communication frequence when encoding. If the shapes of raw
  spaces are [100, 12] and [50, 12], we will get one space with shape [150, 12]
  after fusion.

  The third feature is 0.2, we just return it as a numeric feature.

  The fourth feature is a string, which need to be transformed into an integer
  and then encoded with a continuous space.

  The fifth feature is a multi-value splited by ','. The count of elements is
  not fixed. We need to encode each value into a continuous space and merge
  them together.

  The last feature is a big integer, and just transform it into a continuous
  space.

  All of the above features will be handled by different FeatureColumns, and
  then concatenated by a FeatureGroup.
  """

  def __init__(self):
    pass

  def forward(self, x):
    raise NotImplementedError


class PartitionableColumn(FeatureColumn):
  """ `PartitionableColumn` uses `tf.min_max_variable_partitioner` or
  `tf.fixed_size_partitioner` to partition the embedding varibles.
  Note:
  The `conf.emb_max_partitions` must be provided when using
  partitioner.
  Dynamic Embedding Column must use `tf.fixed_size_partitioner`.
  """
  def _partitioner(self, partitioner='min_max'):
    """
    Args:
      partitioner: 'min_max' or 'fixed'.
    Returns:
      A tensorflow Partitioner or None.
    """
    max_parts = conf.emb_max_partitions
    if max_parts is not None:
      if partitioner == 'min_max':
        return tf.min_max_variable_partitioner(
            max_partitions=max_parts, min_slice_size=conf.emb_min_slice_size)
      else:
        return tf.fixed_size_partitioner(num_shards=max_parts)
    else:
      return None


class NumericColumn(FeatureColumn):
  """ Represents real valued or numerical features.
  Args:
    name: A unique string identifying the input feature.
    normalizer_func: If not `None`, a function that can be used to normalize 
      the value of the tensor. Normalizer function takes the input `Tensor` 
      as its  argument, and returns the output `Tensor`. 
      (e.g. lambda x: (x - 1.0) / 2.0). 
  """
  def __init__(self, name, normalizer_func=None):
    self.normalizer_func = normalizer_func

  def forward(self, x):
    """
    Args:
      x: A 1D Tensor with type tf.float32 or other type which can be casted to 
      tf.float32.
    Returns:
      A `tf.Tensor` with the same shape of the input feature and with the type
      tf.float32.
    """
    if self.normalizer_func is not None:
      x = self.normalizer_func(x)
    x = tf.cast(x, tf.float32)
    return x


class EmbeddingColumn(PartitionableColumn):
  """ Uses embedding_lookup to embed the categorical features.
  Args:
    name: A unique string identifying the input feature.
    bucket_size: The size of the embedding variable.
    dimension: The dimension of the embedding.
    need_hash: Whether need hash the input feature.
  """
  def __init__(self, name, bucket_size, dimension, need_hash=False):
    self._bucket_size = bucket_size
    self._dim = dimension
    self._need_hash = need_hash
    with tf.variable_scope("embedding_column", reuse=tf.AUTO_REUSE):
      self._var = tf.get_variable(
          name=name,
          shape=[bucket_size, dimension],
          partitioner=self._partitioner(partitioner=conf.partitioner),
          trainable=True)

  def forward(self, x):
    """
    Args:
      x: A Tensor of type tf.int64 and shape [batch_size]
    Returns:
      A Tensor of type tf.float32 and shape [batch_size, dimension].
    """
    if self._need_hash:  # int->string->hash
      x = tf.as_string(x)
      x = tf.strings.to_hash_bucket_fast(x, self._bucket_size)
    return tf.nn.embedding_lookup(self._var, x)


class DynamicEmbeddingColumn(PartitionableColumn):
  """ EmbeddingColumn with dynamic bucket_size.
  """
  def __init__(self, name, dimension, is_string=False):
    assert hasattr(tf, "get_embedding_variable"), \
      "Dynamic Embedding Variable is not supported for this tf " \
      "version, you should assign bucket_size for discrete attributes " \
      "in {}.".format(name)
    self._dim = dimension
    self._is_string = is_string
    with tf.variable_scope("dynamic_embedding_column", reuse=tf.AUTO_REUSE):
      self._var = tf.get_embedding_variable(
          name=name,
          embedding_dim=dimension,
          key_dtype=tf.int64,
          partitioner=self._partitioner(partitioner='fixed'),
          trainable=True,
          steps_to_live=conf.emb_live_steps)

  def forward(self, x):
    """
    Args:
      x: A Tensor of type tf.int64 and shape [batch_size]
    Returns:
      A Tensor of type tf.float32 and shape [batch_size, dimension].
    """
    if self._is_string:
      x = tf.strings.to_hash_bucket_fast(
        x, np.iinfo(tf.int64.as_numpy_dtype).max)
    return tf.nn.embedding_lookup(self._var, x)


class FusedEmbeddingColumn(PartitionableColumn):
  """ Fuses the input feature with the same dimension setting and then
  lookups embeddings.
  Args:
    name: A unique string identifying the input feature.
    bucket_list: A list of the size of the embedding variable.
    dimension: The dimension of the embedding.
  """
  def __init__(self, name, bucket_list, dimension):
    self._n = len(bucket_list)
    self._dim = dimension

    self._offsets = []
    offset = 0
    for bucket in bucket_list:
      self._offsets.append(offset)
      offset += bucket

    with tf.variable_scope("fused_embedding_column", reuse=tf.AUTO_REUSE):
      self._var = tf.get_variable(
          name=name,
          shape=[offset, dimension],
          partitioner=self._partitioner(partitioner=conf.partitioner),
          trainable=True)

  def forward(self, x):
    """
    Args:
      x: A list of Tensors of type tf.int64 and shape [batch_size]
    Returns:
      A Tensor of type tf.float32 and shape [batch_size, dimension*len(x)].
    """
    if isinstance(x, list) and len(x) != self._n:
      raise ValueError("{} inputs do not match {} fused columns"
                       .format(len(x), self._n))
    trans_x_list = [(x[i] + self._offsets[i]) for i in range(self._n)]

    if len(trans_x_list[0].shape) == 0:
      x = tf.stack(trans_x_list)
      is_scala = True
    else:
      x = tf.concat(trans_x_list, axis=0)
      is_scala = False

    emb = tf.nn.embedding_lookup(self._var, x)
    emb_list = tf.split(emb, self._n, axis=0)
    ret = tf.concat(emb_list, axis=-1)
    return ret if not is_scala else tf.squeeze(ret, axis=0)


class SparseEmbeddingColumn(PartitionableColumn):
  """ Uses sparse_embedding_lookup to embed the multivalent categorical 
  feature which is split with delimiter.
  Args:
    name: A unique string identifying the input feature.
    bucket_size: The size of the embedding variable.
    dimension: The dimension of the embedding.
    delimiter: The delimiter of multivalent feature.
  """
  def __init__(self, name, bucket_size, dimension, delimiter):
    self._bucket_size = bucket_size
    self._dim = dimension
    self._delimiter = delimiter
    with tf.variable_scope("sparse_embedding_column", reuse=tf.AUTO_REUSE):
      self._var = tf.get_variable(
          name=name,
          shape=[bucket_size, dimension],
          partitioner=self._partitioner(partitioner=conf.partitioner),
          trainable=True)

  def forward(self, x):
    """
    Args:
      x: A Tensor of type tf.int64 and shape [batch_size]
    Returns:
      A Tensor of type tf.float32 and shape [batch_size, dimension].
    """
    is_scala = False
    if len(x.shape) == 0:
      x = tf.expand_dims(x, axis=0)
      is_scala = True

    sparse_x = tf.string_split(x, self._delimiter)
    x = tf.strings.to_hash_bucket_fast(sparse_x.values, self._bucket_size)
    sp = tf.sparse.SparseTensor(sparse_x.indices, x, sparse_x.dense_shape)
    ret = tf.nn.embedding_lookup_sparse(self._var, sp, None)
    return ret if not is_scala else tf.squeeze(ret, axis=0)


class DynamicSparseEmbeddingColumn(PartitionableColumn):
  """ SparseEmbeddingColumn with dynamic bucket_size.
  """
  def __init__(self, name, dimension, delimiter):
    self._dim = dimension
    self._delimiter = delimiter
    assert hasattr(tf, "get_embedding_variable"), \
      "Dynamic Embedding Variable is not supported for this tf " \
      "version, you should assign bucket_size for discrete attributes " \
      "in {}.".format(name)
    with tf.variable_scope("sparse_dynamic_embedding_column", reuse=tf.AUTO_REUSE):
      self._var = tf.get_embedding_variable(
          name=name,
          embedding_dim=dimension,
          key_dtype=tf.int64,
          partitioner=self._partitioner(partitioner='fixed'),
          trainable=True)

  def forward(self, x):
    is_scala = False
    if len(x.shape) == 0:
      x = tf.expand_dims(x, axis=0)
      is_scala = True

    sparse_x = tf.string_split(x, self._delimiter)
    x = tf.strings.to_hash_bucket_fast(
      sparse_x.values, np.iinfo(tf.int64.as_numpy_dtype).max)
    sp = tf.sparse.SparseTensor(sparse_x.indices, x, sparse_x.dense_shape)
    ret = tf.nn.embedding_lookup_sparse(self._var, sp, None)
    return ret if not is_scala else tf.squeeze(ret, axis=0)
