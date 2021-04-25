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

class EgoGINLayerGroup(Module):
  def __init__(self, layers):
    super(EgoGINLayerGroup, self).__init__()
    self.layers = layers

  def append(self, layer):
    self.layers.append(layer)

  def forward(self, x_list, expands):
    """ Compute node embeddings based on GIN.

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

class EgoGINLayer(Module):
  """ GIN. https://arxiv.org/abs/1810.00826.

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
               eps=0.0,
               use_bias=False,
               **kwargs):
    super(EgoGINLayer, self).__init__()

    self.eps = eps
    self.out_dim = output_dim

    if isinstance(input_dim, list) or isinstance(input_dim, tuple):
      self.in_dim = input_dim
      assert len(self.in_dim) == 2
    else:
      self.in_dim = [input_dim, input_dim]

    self.trans = []
    with tf.variable_scope("ego_gin_layer_" + name, reuse=tf.AUTO_REUSE):
      if self.in_dim[0] == self.in_dim[1]:
        self.output = LinearLayer(
            "output", self.in_dim[0], self.out_dim, use_bias)
      else:
        self.output = LinearLayer(
            "output", self.out_dim, self.out_dim, use_bias)
        self.trans.append(
            LinearLayer("trans_x", self.in_dim[0], self.out_dim, use_bias))
        self.trans.append(
            LinearLayer("trans_nbrs", self.in_dim[1], self.out_dim, use_bias))

  def forward(self, x, neighbor, expand):
    """ Compute node embeddings based on GIN.
    ```x_i = W * [(1 + eps) * x_i + sum(x_j) for x_j in N_i]```,
    where ```N_i``` is the neighbor set of ```x_i```.

    Args:
      x: A float tensor with shape = [batch_size, input_dim].
      neighbor: A float tensor with shape = [batch_size * expand, input_dim].
      expand: An integer, the neighbor count.

    Return:
      A float tensor with shape=[batch_size, output_dim].
    """
    nbr = tf.reshape(neighbor, [-1, expand, self.in_dim[1]])
    agg = tf.math.reduce_sum(nbr, axis=1)

    if self.trans:
      x = self.trans[0].forward((1.0 + self.eps) * x)
      agg = self.trans[1].forward(agg)

    return self.output.forward(x + agg)
