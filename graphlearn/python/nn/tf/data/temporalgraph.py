# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
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
from graphlearn.python.nn.tf.data.feature_handler import FeatureHandler


class TemporalGraph(object):
  """ `TemporalGraph` is an expanding of `EgoGraph`, which is used to describe
  a sampled graph with timestamps. It contains the nodes on the events (edges)
  generated according to the time sequence, and the neighbors of the nodes with
  multi-hop fixed expansion factors. The nodes contain the features and times
  of the nodes (typically the time span from the event); the neighbors contain
  the features of the neighboring edges, the times of the edges, and the features
  of the neighboring nodes.

  Args:
    src: A `Data`/Tensor object used to describe the centric nodes.
    src_t: A `Data`/Tensor object used to describe the centric nodes times.
    nbr_nodes: A list of `Data`/Tensor to describe neighborhood nodes.
    nbr_t: A list of `Data`/Tensor to describe neighborhood edges times.
    nbr_edges: A list of `Data`/Tensor to describe neighborhood edges.
    nbr_nums: A list of number of neighbor nodes per hop.
    node_schema: A list of tuple to describe the FeatureSpec of src and
      neighbor nodes. Each tuple is formatted with (name, spec), in which `name`
      is node's type, and `spec` is a FeatureSpec object. Be sure that
      `len(node_schema) == len(neighbors) + 1`.
    edge_schema: A list of tuple to describe the `FeatureSpec` of neighbor edges.
    time_dim: Int, the output dimension of `TimeEncoder` which encode the
      time spans.
  """
  def __init__(self,
               src,
               src_t,
               nbr_nodes,
               nbr_t,
               nbr_edges,
               nbr_nums,
               node_schema,
               edge_schema,
               time_dim):
    self._src = src
    self._src_t = src_t
    self._nbr_nodes = nbr_nodes
    self._nbr_t = nbr_t
    self._nbr_edges = nbr_edges
    self._nbr_nums = nbr_nums
    self._node_schema = node_schema
    self._edge_schema = edge_schema
    self._time_dim = time_dim

  def transform(self):
    def transform_feat(feat, schema):
      feat_handler = FeatureHandler(schema[0], schema[1])
      return feat_handler.forward(feat)
    src = transform_feat(self.src, self.node_schema[0])

    src_t = self._encode_time(self.src_t)

    nbr_nodes = []
    for nbr, schema in zip(self.nbr_nodes, self.node_schema[1:]):
      nbr_nodes.append(transform_feat(nbr, schema))

    nbr_t = [self._encode_time(t) for t in self.nbr_t]

    nbr_edges = []
    if self.nbr_edges:
      assert len(self.edge_schema) == len(self.nbr_edges)
      for e, schema in zip(self.nbr_edges, self.edge_schema):
        nbr_edges.append(transform_feat(e, schema))

    return TemporalGraph(src, src_t, nbr_nodes, nbr_t, nbr_edges,
                         self.nbr_nums, None, None, self.time_dim)

  def _encode_time(self, t):
    ts_encode = TimeEncoder('time_span', self.time_dim)
    return ts_encode.forward(t)

  @property
  def src(self):
    return self._src

  @property
  def src_t(self):
    return self._src_t

  @property
  def nbr_nodes(self):
    return self._nbr_nodes

  @property
  def nbr_t(self):
    return self._nbr_t

  @property
  def nbr_edges(self):
    return self._nbr_edges

  @property
  def nbr_nums(self):
    return self._nbr_nums

  @property
  def node_schema(self):
    return self._node_schema

  @property
  def edge_schema(self):
    return self._edge_schema

  @property
  def time_dim(self):
    return self._time_dim

  def hop_node(self, i):
    return self.nbr_nodes[i]

  def hop_edge(self, i):
    return self.nbr_edges[i]

  def hop_t(self, i):
    return self.nbr_t[i]

class TimeEncoder(object):
  def __init__(self, name, time_dim):
    super(TimeEncoder, self).__init__()
    self.time_dim = time_dim

    with tf.variable_scope("time_span_column", reuse=tf.AUTO_REUSE):
      init = tf.constant(1 / 10 ** np.linspace(0, 9, time_dim), dtype=tf.float32)
      self.basis_freq = tf.get_variable(name + '_basis_freq', initializer=init)
      self.phase = tf.get_variable(name + '_phase', shape=[time_dim])

  def forward(self, ts):
    ts = tf.cast(ts, tf.float32)
    ts = tf.expand_dims(ts, -1) # [N(batch-size), L(sequence-length)] -> [N, L, 1]
    basis_freq = tf.reshape(self.basis_freq, [1, 1, -1])
    map_ts = tf.multiply(ts, basis_freq)
    phase = tf.reshape(self.phase, [1, 1, -1])
    map_ts = tf.add(map_ts, phase)
    harmonic = tf.math.cos(map_ts)
    return tf.reshape(harmonic, [-1, self.time_dim])
