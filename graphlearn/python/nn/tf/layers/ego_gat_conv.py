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
"""EgoGraph based GAT convolutional layer"""

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

from graphlearn.python.nn.tf.config import conf
from graphlearn.python.nn.tf.layers.ego_layer import EgoConv, EgoLayer
from graphlearn.python.nn.tf.layers.linear_layer import LinearLayer


class EgoGATConv(EgoConv):
  """ Graph Attention Network. https://arxiv.org/pdf/1710.10903.pdf.

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
    use_bias: A boolean, whether add bias after computation.
  """

  def __init__(self,
               name,
               in_dim,
               out_dim,
               num_head=1,
               use_bias=False,
               attn_dropout=0.0,
               **kwargs):
    super(EgoGATConv, self).__init__()

    self._out_dim = out_dim
    self._num_head = num_head
    self._attn_dropout = attn_dropout

    is_homo = False # src and nbr with different input dimension.
    if isinstance(in_dim, list) or isinstance(in_dim, tuple):
      self._in_dim = in_dim
      assert len(self._in_dim) == 2
    else:
      self._in_dim = [in_dim, in_dim]
      is_homo = True

    self.att_layers, self.linear_xs, self.linear_ns = [], [], []
    with tf.variable_scope("ego_gat_layer_" + name, reuse=tf.AUTO_REUSE):
      for i in range(num_head):
        layers = self.build_linear_layers(i, is_homo, use_bias)
        self.att_layers.append(layers[0])
        self.linear_xs.append(layers[1])
        self.linear_ns.append(layers[2])

  def build_linear_layers(self, i, is_homo, use_bias):
    attn_layer = LinearLayer(
        "attn_" + str(i), self._out_dim * 2, 1, use_bias)
    x_layer = LinearLayer(
        "x_" + str(i), self._in_dim[0], self._out_dim, use_bias)
    n_layer = x_layer if is_homo else LinearLayer(
        "n_" + str(i), self._in_dim[1], self._out_dim, use_bias)
    return attn_layer, x_layer, n_layer

  def singel_head_conv(self, x, neighbor, expand, i):
    x = self.linear_xs[i](x)               # [batch_size, out_dim]
    neighbor = self.linear_ns[i](neighbor) # [batch_size * expand, out_dim]

    x = tf.expand_dims(x, 1)               # [batch_size, 1, out_dim]
    x = tf.tile(x, [1, expand, 1])         # [batch_size, expand, out_dim]
    neighbor = tf.reshape(neighbor, [-1, expand, self._out_dim])
    x_neighbor = tf.concat([x, neighbor], axis=-1)
    x_neighbor = tf.reshape(x_neighbor, [-1, 2 * self._out_dim])
    coefficients = self.att_layers[i](x_neighbor)  # [batch_size * expand, 1]
    coefficients = tf.reshape(coefficients, [-1, expand, 1])  # [batch_size, expand, 1]
    coefficients = tf.nn.softmax(tf.nn.leaky_relu(coefficients), axis=1)
    if self._attn_dropout and conf.training:
      coefficients = tf.nn.dropout(coefficients, 1 - self._attn_dropout)
    neighbor = tf.multiply(coefficients, neighbor) # [batch_size, expand, out_dim]
    return tf.reduce_sum(neighbor, axis=1)         # [batch_size, out_dim]

  def forward(self, x, neighbor, expand):
    """ Compute node embeddings based on GAT.
    Args:
      x: A float tensor with shape = [batch_size, in_dim].
      neighbor: A float tensor with shape = [batch_size * expand, in_dim].
      expand: An integer, the neighbor count.

    Return:
      A float tensor with shape=[batch_size, out_dim].
    """
    rets = [self.singel_head_conv(x, neighbor, expand, i) for i in range(self._num_head)]
    return tf.reduce_mean(rets, axis=0)