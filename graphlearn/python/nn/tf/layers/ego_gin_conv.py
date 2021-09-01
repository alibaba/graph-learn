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
"""EgoGraph based GIN convolutional layer"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from graphlearn.python.nn.tf.layers.ego_layer import EgoConv, EgoLayer
from graphlearn.python.nn.tf.layers.linear_layer import LinearLayer


class EgoGINConv(EgoConv):
  """ GIN. https://arxiv.org/abs/1810.00826.

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
               eps=0.0,
               use_bias=False,
               **kwargs):
    super(EgoGINConv, self).__init__()

    self._eps = eps
    self._out_dim = out_dim

    if isinstance(in_dim, list) or isinstance(in_dim, tuple):
      self._in_dim = in_dim
      assert len(self._in_dim) == 2
    else:
      self._in_dim = [in_dim, in_dim]

    self.trans = []
    with tf.variable_scope("ego_gin_layer_" + name, reuse=tf.AUTO_REUSE):
      if self._in_dim[0] == self._in_dim[1]:
        self.output = LinearLayer(
            "output", self._in_dim[0], self._out_dim, use_bias)
      else:
        self.output = LinearLayer(
            "output", self._out_dim, self._out_dim, use_bias)
        self.trans.append(
            LinearLayer("trans_x", self._in_dim[0], self._out_dim, use_bias))
        self.trans.append(
            LinearLayer("trans_nbrs", self._in_dim[1], self._out_dim, use_bias))

  def forward(self, x, neighbor, expand):
    """ Compute node embeddings based on GIN.
    ```x_i = W * [(1 + eps) * x_i + sum(x_j) for x_j in N_i]```,
    where ```N_i``` is the neighbor set of ```x_i```.

    Args:
      x: A float tensor with shape = [batch_size, in_dim].
      neighbor: A float tensor with shape = [batch_size * expand, in_dim].
      expand: An integer, the neighbor count.

    Return:
      A float tensor with shape=[batch_size, out_dim].
    """
    nbr = tf.reshape(neighbor, [-1, expand, self._in_dim[1]])
    agg = tf.math.reduce_sum(nbr, axis=1)

    if self.trans:
      x = self.trans[0].forward((1.0 + self._eps) * x)
      agg = self.trans[1].forward(agg)

    return self.output.forward(x + agg)
