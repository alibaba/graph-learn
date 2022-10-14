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
"""EgoGraph based RGCN convolutional layer"""

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

from graphlearn.python.nn.tf.layers.ego_layer import EgoConv, EgoLayer
from graphlearn.python.nn.tf.layers.linear_layer import LinearLayer


class EgoRGCNConv(EgoConv):
  """ EgoGraph based implementation of RGCN. https://arxiv.org/abs/1703.06103.

  Args:
    name: A string, layer name.
    in_dim: An integer or a two elements tuple. Dimension of input features.
      If an integer, nodes and neighbors share the same dimension.
      If an tuple, the two elements represent the dimensions of node features
      and neighbor features.
      Usually, different dimensions happen in the heterogeneous graph.
    out_dim: An integer, dimension of the output embeddings. Both the node
      features and neighbor features will be encoded into the same dimension,
      and then do some combination.
    num_relations: Number of relations.
    num_bases: Denotes the number of bases to use, default is None.
    num_blocks: Denotes the number of blocks to use, default is None.
      Note that num_bases and num_blocks cannot be set at the same time.
    agg_type: A string, how to merge neighbor values. The optional values are
      'mean', 'sum', 'max'.
    use_bias: A boolean, whether add bias after computation.
  """

  def __init__(self,
               name,
               in_dim,
               out_dim,
               num_relations,
               num_bases=None,
               num_blocks=None,
               agg_type="mean",
               use_bias=False,
               **kwargs):
    super(EgoRGCNConv, self).__init__()
    assert agg_type in {"mean", "sum", "max"}
    if num_bases is not None and num_blocks is not None:
      raise ValueError('Can not apply both basis- and block-diagonal-decomposition '
                       'regularization at the same time.')

    self._agg_type = agg_type
    self._out_dim = out_dim
    self._num_relations = num_relations
    self._num_bases = num_bases
    self._num_blocks = num_blocks

    if isinstance(in_dim, list) or isinstance(in_dim, tuple):
      self._in_dim = in_dim
      assert len(self._in_dim) == 2
    else:
      self._in_dim = [in_dim, in_dim]

    with tf.variable_scope("ego_rgcn_layer_" + name, reuse=tf.AUTO_REUSE):
      self.root_weight = LinearLayer("root_weight", self._in_dim[0], self._out_dim)
      # neighbor's weight
      if num_bases is not None:
        self.weight = tf.get_variable(
          name="weight",
          shape=[num_bases, self._in_dim[1], out_dim])
        self.coefficient = tf.get_variable(
          name="coefficient",
          shape=[num_relations, num_bases])
      elif num_blocks is not None:
        assert (self._in_dim[1] % num_blocks == 0 and out_dim % num_blocks == 0)
        self.weight = tf.get_variable(
          name='weight',
          shape=[num_relations, num_blocks, self._in_dim[1] // num_blocks, out_dim // num_blocks])
      else:
        self.weight = tf.get_variable(
          name="weight", 
          shape=[num_relations, self._in_dim[1], self._out_dim])
      
      if use_bias:
        self.bias = tf.get_variable(name="bias", shape=[out_dim])
      else:
        self.bias = None

  def forward(self, x, neighbor, expand):
    """
    Args:
      x: A float tensor with shape = [batch_size, input_dim].
      neighbor: A list of float tensors with shape = [batch_size * expand, input_dim].
      expand: An integer, the neighbor count.
    """
    agg_func = self.aggregator()
    weight = self.weight
    out = []
    if self._num_bases is not None:
      weight = tf.reshape(self.weight, [self._num_bases, -1])
      weight = tf.reshape(tf.matmul(self.coefficient, weight), 
                          [self._num_relations, self._in_dim[1], self._out_dim])

    if self._num_blocks is not None:
      for i in range(self._num_relations):
        h = agg_func(neighbor[i], expand, self._in_dim[1])
        h = tf.reshape(h, [-1, self._num_blocks, self._in_dim[1] // self._num_blocks])
        h = tf.einsum('aij,ijk->aik', h, weight[i])
        out.append(tf.reshape(h, [-1, self._out_dim]))
    else:
      for i in range(self._num_relations):
        h = agg_func(neighbor[i], expand, self._in_dim[1])
        h = tf.matmul(h, weight[i])
        out.append(h)
    out = tf.math.add_n(out)

    out += self.root_weight(x)
    if self.bias is not None:
      out += self.bias
    return out

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