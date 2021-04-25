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

import tensorflow as tf
from graphlearn.python.nn.tf.module import Module
from graphlearn.python.nn.tf.layers.linear_layer import LinearLayer

class GINLayer(Module):
  """ GIN. https://arxiv.org/abs/1810.00826.

  Args:
    name: A string, layer name.
    input_dim: An integer, dimension of input features.
    output_dim: An integer, dimension of output embeddings.
    eps: A float, coefficient offset of centric vertex.
    use_bias: A boolean, whether add bias after computation.
  """

  def __init__(self,
               name,
               input_dim,
               output_dim,
               eps=0.0,
               use_bias=False,
               **kwargs):
    self.eps = eps
    self.linear = LinearLayer(name, input_dim, output_dim, use_bias)

  def forward(self, x, graph):
    """ Compute node embeddings based on GIN.
    ```x_i = W * [(1 + eps) * x_i + sum(x_j) for x_j in N_i]```,
    where ```N_i``` is the neighbor set of ```x_i```.

    Args:
      x: Input node vectors, a tensor with shape [n, input_dim]. `n` is the
        count of nodes, and `input_dim` is feature dimension which must be the
        same with layers[0].
      graph: SubGraph object, including adjacent matrix and node degrees info.
        The `graph.A` is a tensor with shape [n, n]. Each element of `A`
        indicates the weight between two nodes. e.g. A[i, j] means the edge
        weight between x[i] and x[j].

    Return:
      A tensor with shape [n, output_dim].
    """
    # Here we use matmul to compute the sum all neighbors. graph.A is the
    # adjacent matrix. Some kind of sparse operation will be needed if graph.A
    # is described in sparse format.
    agg = tf.matmul(graph.A, x)
    output = (1.0 + self.eps) * x + agg
    return self.linear(output)
