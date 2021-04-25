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
import tensorflow as tf
from graphlearn.python.nn.tf.config import conf
from graphlearn.python.nn.tf.module import Module


class FeatureColumn(Module):
  """ Transform raw features based on configurations and return dense tensors.

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

  The third feature is 0.2, we may just return it as a numeric feature or
  multiply by a weight.

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
    return self.dense(x)

  def dense(self, x):
    return self.transform(x)

  def transfrom(self):
    raise NotImplementedError


class PartitionableColumn(FeatureColumn):
  def _partitioner(self):
    cluster_spec = os.environ.get("CLUSTER_SPEC")
    if not cluster_spec:
      return None
    ps_count = tf.train.ClusterSpec(json.loads(cluster_spec)).num_tasks("ps")
    return tf.min_max_variable_partitioner(
        max_partitions=ps_count, min_slice_size=conf.emb_min_slice_size)


class EmbeddingColumn(PartitionableColumn):
  def __init__(self, name, bucket_size, dimension, need_hash=False):
    self._bucket_size = bucket_size
    self._dim = dimension
    self._need_hash = need_hash
    with tf.variable_scope("embedding_column", reuse=tf.AUTO_REUSE):
      self._var = tf.get_variable(
          name=name,
          shape=[bucket_size, dimension],
          partitioner=self._partitioner(),
          trainable=True)

  def transform(self, x):
    if self._need_hash:  # int->string->hash
      x = tf.as_string(x)
      x = tf.strings.to_hash_bucket_fast(x, self._bucket_size)
    return tf.nn.embedding_lookup(self._var, x)

class DynamicEmbeddingColumn(PartitionableColumn):
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
          partitioner=self._partitioner(),
          trainable=True,
          steps_to_live=conf.emb_live_steps)

  def transform(self, x):
    if self._is_string:
      x = tf.strings.to_hash_bucket_fast(
        x, np.iinfo(tf.int64.as_numpy_dtype).max)
    return tf.nn.embedding_lookup(self._var, x)


class NumericColumn(FeatureColumn):
  def __init__(self, name, weight=1.0, is_float=True):
    self._weight = weight
    self._is_float = is_float

  def transform(self, x):
    if not self._is_float:
      x = tf.cast(x, tf.float32)
    if self._weight != None:
      x = tf.math.multiply(x, self._weight)
    return x

class FusedEmbeddingColumn(PartitionableColumn):
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
          partitioner=self._partitioner(),
          trainable=True)

  def transform(self, x):
    if isinstance(x, list) and len(x) != self._n:
      raise ValueError("{} inputs do not match {} fused columns"
                       .format(len(x), self._n))
    if isinstance(x, np.ndarray) and x.shape[0] != self._n:
      raise ValueError("{} inputs do not match {} fused columns"
                       .format(x.shape[0], self._n))

    trans_x_list = [
        tf.convert_to_tensor(x[i] + self._offsets[i]) for i in range(self._n)]

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
  def __init__(self, name, bucket_size, dimension, delimiter):
    self._bucket_size = bucket_size
    self._dim = dimension
    self._delimiter = delimiter
    with tf.variable_scope("sparse_embedding_column", reuse=tf.AUTO_REUSE):
      self._var = tf.get_variable(
          name=name,
          shape=[bucket_size, dimension],
          partitioner=self._partitioner(),
          trainable=True)

  def transform(self, x):
    x = tf.convert_to_tensor(x)
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
          partitioner=self._partitioner(),
          trainable=True)

  def transform(self, x):
    x = tf.convert_to_tensor(x)
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
