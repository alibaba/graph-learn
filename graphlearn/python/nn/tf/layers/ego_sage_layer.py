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
from graphlearn.python.nn.tf.module import Module
from graphlearn.python.nn.tf.layers.linear_layer import LinearLayer

class EgoSAGELayerGroup(Module):
  def __init__(self, layers):
    super(EgoSAGELayerGroup, self).__init__()
    self.layers = layers

  def append(self, layer):
    self.layers.append(layer)

  def forward(self, x_list, expands):
    """ Compute node embeddings based on GraphSAGE.

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

class EgoSAGELayer(Module):
  """ GraphSAGE. https://arxiv.org/abs/1706.02216.

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
    agg_type: A string, how to merge neighbor values. The optional values are
      'mean', 'sum', 'max'.
    com_type: A string, how to combine neighbors to self. The optional values
      are 'add', 'concat'.
    use_bias: A boolean, whether add bias after computation.
  """

  def __init__(self,
               name,
               input_dim,
               output_dim,
               agg_type="mean",
               com_type="add",
               use_bias=False,
               parameter_share=False,
               **kwargs):
    super(EgoSAGELayer, self).__init__()
    assert agg_type in {"mean", "sum", "max", "gcn"}
    assert com_type in {"add", "concat", "gcn"}

    self.agg_type = agg_type
    self.com_type = com_type
    self.out_dim = output_dim

    if isinstance(input_dim, list) or isinstance(input_dim, tuple):
      self.in_dim = input_dim
      assert len(self.in_dim) == 2
    else:
      self.in_dim = [input_dim, input_dim]

    with tf.variable_scope("ego_sage_layer_" + name, reuse=tf.AUTO_REUSE):
      self.linears = self.add_transform_layers(parameter_share, use_bias)

  def add_transform_layers(self, parameter_share, use_bias):
    layers = []
    if self.com_type == "concat":
      dim = self.in_dim[0] + self.in_dim[1]
      layers.append(LinearLayer("trans_nodes", dim, self.out_dim, use_bias))
    elif parameter_share and self.in_dim[0] == self.in_dim[1]:
      layer = LinearLayer("trans_nodes", self.in_dim[0], self.out_dim, use_bias)
      layers.append(layer)
      layers.append(layer)
    else:
      layers.append(
          LinearLayer("trans_nodes", self.in_dim[0], self.out_dim, use_bias))
      layers.append(
          LinearLayer("trans_nbrs", self.in_dim[1], self.out_dim, use_bias))
    return layers

  def forward(self, x, neighbor, expand):
    """ Compute node embeddings based on GraphSAGE.
    Args:
      x: A float tensor with shape = [batch_size, input_dim].
      neighbor: A float tensor with shape = [batch_size * expand, input_dim].
      expand: An integer, the neighbor count.

    Return:
      A float tensor with shape=[batch_size, output_dim].
    """
    # aggregate neighbors at each hop
    agg_func = self.aggregator()
    agg_info = agg_func(neighbor, expand, self.in_dim[1])

    # combine self info with aggregated neighbors
    comb_func = self.combiner()
    return comb_func(x, agg_info)

  def aggregator(self):
    func_name = self.agg_type + "_agg"
    if not hasattr(self, func_name):
      raise TypeError("Unsupported agg_type: " + self.agg_type)
    return getattr(self, func_name)

  def sum_agg(self, x, expand, dim):
    t = tf.reshape(x, [-1, expand, dim])
    return tf.math.reduce_sum(t, axis=1)

  def mean_agg(self, x, expand, dim):
    t = tf.reshape(x, [-1, expand, dim])
    return tf.math.reduce_mean(t, axis=1)

  def gcn_agg(self, x, expand, dim):
    return tf.reshape(x, [-1, expand, dim])

  def max_agg(self, x, expand, dim):
    t = tf.reshape(x, [-1, expand, dim])
    return tf.math.reduce_max(t, axis=1)

  def combiner(self):
    func_name = self.com_type + "_comb"
    if not hasattr(self, func_name):
      raise TypeError("Unsupported com_type: " + self.com_type)
    return getattr(self, func_name)

  def add_comb(self, x, neighbors):
    current = self.linears[0](x)
    nbr_info = self.linears[1](neighbors)
    return tf.add(current, nbr_info)

  def concat_comb(self, x, neighbors):
    tmp = tf.concat([x, neighbors], axis=1)
    ret = self.linears[0](tmp)
    return ret

  def gcn_comb(self, x, neighbors):
    x = tf.reduce_mean(tf.concat(
          [neighbors, tf.expand_dims(x, axis=1)], axis=1), axis=1)
    return self.linears[0](x)
