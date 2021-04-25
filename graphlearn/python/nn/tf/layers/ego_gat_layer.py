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

import numpy as np
import tensorflow as tf
from graphlearn.python.nn.tf.config import conf
from graphlearn.python.nn.tf.module import Module
from graphlearn.python.nn.tf.layers.linear_layer import LinearLayer

class EgoGATLayerGroup(Module):
  def __init__(self, layers):
    super(EgoGATLayerGroup, self).__init__()
    self.layers = layers

  def append(self, layer):
    self.layers.append(layer)

  def forward(self, x_list, expands):
    """ Compute node embeddings based on GAT.

    x_list = [nodes, hop1, hop2, ... , hopK-1, hopK]
               |   /  |   /  |   /        |    /
               |  /   |  /   |  /         |   /
               | /    | /    | /          |  /
    output = [ret0,  ret1, ret2, ... , retK-1]

    Args:
      x_list: A list of tensors, representing input nodes and their neighbors.
        If len(x_list) is K+1, that means x_list[0], x_list[1], ... , x_list[K]
        are the hidden embedding values at each hop. Tensors in x_list[i] are
        the neighbors of that in x_list[i-1]. In this layer, we will do
        convolution for each adjencent pair and return a list with length K.

        The shape of x_list[0] is `[n, input_dim_0]`, and the shape of x_list[i]
        is `[n * k_1 * ... * k_i, input_dim_i]`, where `k_i` means the neighbor
        count of each node at (i-1)th hop. Each `input_dim_i` must match with
        `input_dim` parameter when layer construction.

      expands: An integer list of neighbor count at each hop. For the above
        x_list, expands = [k_1, k_2, ... , k_K]

    Return:
      A list with K tensors, and the ith shape is
      `[n * k_1 * ... * k_i, output_dim]`.
    """
    assert len(self.layers) == (len(x_list) - 1)
    assert len(self.layers) == len(expands)

    rets = []
    for i in range(1, len(x_list)):
      x = x_list[i - 1]
      neighbors = x_list[i]
      ret = self.layers[i - 1](x, neighbors, expands[i - 1])
      rets.append(ret)
    return rets

class EgoGATLayer(Module):
  """ Graph Attention Network. https://arxiv.org/pdf/1710.10903.pdf.

  Args:
    name: A string, layer name.
    input_dim: An integer or a two elements tuple. Dimension of input features.
      If an integer, nodes and neighbors share the same dimension.
      If an tuple, the two elements represent the dimensions of node features
      and neighbor features.
      Usually, different dimensions happen in the heterogeneous graph.
    output_dim: An integer, dimension of the output embeddings. Both the node
      features and neighbor features will be encoded into the same dimension,
      and then do some combination.
    use_bias: A boolean, whether add bias after computation.
  """

  def __init__(self,
               name,
               input_dim,
               output_dim,
               num_head=1,
               use_bias=False,
               attn_dropout=None,
               **kwargs):
    super(EgoGATLayer, self).__init__()

    self.out_dim = output_dim
    self.num_head = num_head
    self.attn_dropout = attn_dropout

    is_homo = False
    if isinstance(input_dim, list) or isinstance(input_dim, tuple):
      self.in_dim = input_dim
      assert len(self.in_dim) == 2
    else:
      self.in_dim = [input_dim, input_dim]
      is_homo = True

    self.att_layers, self.linear_xs, self.linear_ns = [], [], []
    with tf.variable_scope("ego_gat_layer_" + name, reuse=tf.AUTO_REUSE):
      for i in range(num_head):
        layers = self.get_layers(i, is_homo, use_bias)
        self.att_layers.append(layers[0])
        self.linear_xs.append(layers[1])
        self.linear_ns.append(layers[2])

  def get_layers(self, i, is_homo, use_bias):
    attn_layer = LinearLayer(
        "attn_" + str(i), self.out_dim * 2, 1, use_bias)
    x_layer = LinearLayer(
        "x_" + str(i), self.in_dim[0], self.out_dim, use_bias)
    n_layer = x_layer if is_homo else LinearLayer(
        "n_" + str(i), self.in_dim[1], self.out_dim, use_bias)
    return attn_layer, x_layer, n_layer

  def do(self, x, neighbor, expand, i):
    x = self.linear_xs[i](x)               # [batch_size, out_dim]
    neighbor = self.linear_ns[i](neighbor) # [batch_size * expand, out_dim]

    x = tf.expand_dims(x, 1)               # [batch_size, 1, out_dim]
    x = tf.tile(x, [1, expand, 1])         # [batch_size, expand, out_dim]
    neighbor = tf.reshape(neighbor, [-1, expand, self.out_dim])
    x_neighbor = tf.concat([x, neighbor], axis=-1)
    x_neighbor = tf.reshape(x_neighbor, [-1, 2 * self.out_dim])
    coefficients = self.att_layers[i](x_neighbor)  # [batch_size * expand, 1]
    coefficients = tf.reshape(coefficients, [-1, expand, 1])  # [batch_size, expand, 1]
    coefficients = tf.nn.softmax(tf.nn.leaky_relu(coefficients), axis=1)
    if self.attn_dropout is not None and conf.training:
      coefficients = tf.nn.dropout(coefficients, 1 - self.attn_dropout)
    neighbor = tf.multiply(coefficients, neighbor) # [batch_size, expand, out_dim]
    return tf.reduce_sum(neighbor, axis=1)         # [batch_size, output_dim]

  def forward(self, x, neighbor, expand):
    """ Compute node embeddings based on GAT.
    Args:
      x: A float tensor with shape = [batch_size, input_dim].
      neighbor: A float tensor with shape = [batch_size * expand, input_dim].
      expand: An integer, the neighbor count.

    Return:
      A float tensor with shape=[batch_size, output_dim].
    """
    rets = [self.do(x, neighbor, expand, i) for i in range(self.num_head)]
    return tf.reduce_mean(rets, axis=0)
