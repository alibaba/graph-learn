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
"""EgoGraph based GraphSAGE convolutional layer"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from graphlearn.python.nn.tf.layers.ego_layer import EgoConv, EgoLayer
from graphlearn.python.nn.tf.layers.linear_layer import LinearLayer


class EgoSAGEConv(EgoConv):
  """ GraphSAGE. https://arxiv.org/abs/1706.02216.

  Args:
    name: A string, layer name.
    in_dim: An integer or a two elements tuple. Dimension of input features.
      If an integer, nodes and neighbors share the same dimension.
      If an tuple, the two elements represent the dimensions of node features
      and neighbor features.
      Usually, different dimensions happen in the heterogeneous graph. Note that
      for 'gcn' agg_type, in_dim must be an interger cause gcn is only for 
      homogeneous graph.
    out_dim: An integer, dimension of the output embeddings. Both the node
      features and neighbor features will be encoded into the same dimension,
      and then do some combination.
    agg_type: A string, how to merge neighbor values. The optional values are
      'mean', 'sum', 'max' and 'gcn'.
    use_bias: A boolean, whether add bias after computation.
  """

  def __init__(self,
               name,
               in_dim,
               out_dim,
               agg_type="mean",
               use_bias=False,
               **kwargs):
    super(EgoSAGEConv, self).__init__()
    assert agg_type in {"mean", "sum", "max", "gcn"}
    self._agg_type = agg_type
    self._out_dim = out_dim
    if isinstance(in_dim, list) or isinstance(in_dim, tuple):
      self._in_dim = in_dim
      assert len(self._in_dim) == 2
      assert self._agg_type != 'gcn'
    else:
      self._in_dim = [in_dim, in_dim]

    with tf.variable_scope("ego_sage_layer_" + name, reuse=tf.AUTO_REUSE):
      dim = self._in_dim[0] if self._agg_type == 'gcn' else self._in_dim[0] + self._in_dim[1]
      self.linear = LinearLayer("trans_nodes", dim, self._out_dim, use_bias)

  def forward(self, x, neighbor, expand):
    # aggregate
    agg_func = self.aggregator()
    agg_nbr = agg_func(neighbor, expand, self._in_dim[1])
    # update
    if self._agg_type == 'gcn':
      x = tf.reduce_mean(tf.concat(
        [agg_nbr, tf.expand_dims(x, axis=1)], axis=1), axis=1)
      return self.linear(x)
    else:
      return self.linear(tf.concat([x, agg_nbr], axis=1))

  def aggregator(self):
    func_name = self._agg_type + "_agg"
    if not hasattr(self, func_name):
      raise TypeError("Unsupported agg_type: " + self._agg_type)
    return getattr(self, func_name)

  def sum_agg(self, x, expand, dim):
    t = tf.reshape(x, [-1, expand, dim])
    return tf.math.reduce_sum(t, axis=1)

  def mean_agg(self, x, expand, dim):
    t = tf.reshape(x, [-1, expand, dim])
    return tf.math.reduce_mean(t, axis=1)

  def max_agg(self, x, expand, dim):
    t = tf.reshape(x, [-1, expand, dim])
    return tf.math.reduce_max(t, axis=1)
  
  def gcn_agg(self, x, expand, dim):
    return tf.reshape(x, [-1, expand, dim])