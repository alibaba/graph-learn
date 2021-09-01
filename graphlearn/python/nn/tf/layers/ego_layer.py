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
"""base classes of EgoGraph based graph convolutional layer"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
from graphlearn.python.nn.tf.module import Module


class EgoConv(Module):
  """Represents the single propagation of 1-hop neighbor to centeric nodes."""
  __metaclass__ = ABCMeta

  @abstractmethod
  def forward(self, x, neighbor, expand):
    """ Update centeric node embeddings by aggregating neighbors.
    Args:
      x: A float tensor with shape = [batch_size, input_dim].
      neighbor: A float tensor with shape = [batch_size * expand, input_dim].
      expand: An integer, the neighbor count.

    Return:
      A float tensor with shape=[batch_size, output_dim].
    """


class EgoLayer(Module):
  """Denotes one convolution of all nodes on the `EgoGraph`. 
  For heterogeneous graphs, there are different types of nodes and edges, so 
  one convolution process of graph may contain serveral different aggregations
  of 1-hop neighbors based on node type and edge type. We denote `EgoConv` as 
  a single propogation of 1-hop neighbor to centric nodes, and use `EgoLayer` to
  represent the entire 1-hop propogation of `EgoGraph`.
  """
  def __init__(self, convs):
    super(EgoLayer, self).__init__()
    self.convs = convs

  def forward(self, x_list, expands):
    """ Update node embeddings.

    x_list = [nodes, hop1, hop2, ... , hopK-1, hopK]
               |   /  |   /  |   /        |    /
               |  /   |  /   |  /         |   /
               | /    | /    | /          |  /
    output = [ret0,  ret1, ret2, ... , retK-1]

    Args:
      x_list: A list of tensors, representing input nodes and their K-hop neighbors.
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
    assert len(self.convs) == (len(x_list) - 1)
    assert len(self.convs) == len(expands)

    rets = []
    for i in range(1, len(x_list)):
      x = x_list[i - 1]
      neighbors = x_list[i]
      ret = self.convs[i - 1](x, neighbors, expands[i - 1])
      rets.append(ret)
    return rets


  def append(self, conv):
    self.convs.append(conv)