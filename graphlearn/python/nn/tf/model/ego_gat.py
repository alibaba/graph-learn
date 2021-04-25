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
from graphlearn.python.nn.tf.config import conf
from graphlearn.python.nn.tf.module import Module
from graphlearn.python.nn.tf.layers.ego_gat_layer import EgoGATLayer
from graphlearn.python.nn.tf.layers.ego_gat_layer import EgoGATLayerGroup

class EgoGAT(Module):
  """ Graph Attention Network. https://arxiv.org/pdf/1710.10903.pdf.

  Args:
    layers: A list, each element is an `EgoGATLayerGroup`.
    bn_fn: Batch normalization function for hidden layers' output. Default is
      None, which means batch normalization will not be performed.
    active_fn: Activation function for hidden layers' output. Default is None,
      which means activation will not be performed.
    dropout: Dropout rate for hidden layers' output. Default is None, which
      means dropout will not be performed. The optional value is a float.
  """

  def __init__(self,
               layers,
               bn_fn=None,
               active_fn=None,
               dropout=None,
               **kwargs):
    super(EgoGAT, self).__init__()

    self.layers = layers
    self.bn_func = bn_fn
    self.active_func = active_fn

    if dropout is not None:
      self.dropout_func = lambda x: tf.nn.dropout(x, keep_prob=1-dropout)
    else:
      self.dropout_func = None

  def forward(self, graph):
    """ Compute node embeddings through the GAT layers.

    h^{i} is a list, 0 <= i <= n, where n is len(layers).
    h^{i} = [ h_{0}^{i}, h_{1}^{i}, h_{2}^{i}, ... , h_{n - i}^{i} ]

    For 3 layers, we need nodes and 3-hop neighbors in the graph object. And
      h^{0} = [ h_{0}^{0}, h_{1}^{0}, h_{2}^{0}, h_{3}^{0} ]
      h^{1} = [ h_{0}^{1}, h_{1}^{1}, h_{2}^{1} ]
      h^{2} = [ h_{0}^{2}, h_{1}^{2} ]
      h^{3} = [ h_{0}^{3} ]

    For initialization,
      h_{0}^{0} = graph.nodes
      h_{1}^{0} = graph.hop_{1}
      h_{2}^{0} = graph.hop_{2}
      h_{3}^{0} = graph.hop_{3}

    And then h^{i} = layer_{i}(h^{i-1}), h_{0}^{3} is our returned value.

    Args:
      graph: EgoGraph object. An EgoGraph contains a batch of nodes and their
        n-hop neighbors. In EgoGraph, wo do not care where the neighbors come
        from, a homogeneous graph, or a heterogeneous one.

    Return:
      A tensor with shape [batch_size, output_dim], where `output_dim` is the
      same with layers[-1].
    """
    graph = graph.forward()

    # h^{0}
    h = [graph.nodes]
    for i in range(len(self.layers)):
      h.append(graph.hop(i))

    hops = graph.expands
    for i in range(len(self.layers) - 1):
      # h^{i}
      current_hops = hops if i == 0 else hops[:-i]
      h = self.layers[i].forward(h, current_hops)
      H = []
      for x in h:
        if self.bn_func:
          x = self.bn_func(x)
        if self.active_func:
          x = self.active_func(x)
        if self.dropout_func and conf.training:
          x = self.dropout_func(x)
        H.append(x)
      h = H

    # The last layer
    h = self.layers[-1].forward(h, [hops[0]])
    assert len(h) == 1
    return h[0]

class HomoEgoGAT(EgoGAT):
  def __init__(self,
               dims,
               num_head=1,
               bn_fn=None,
               active_fn=None,
               dropout=None,
               attn_dropout=None,
               **kwargs):
    """ In a homogeneous graph, all the nodes are of the same type, due to which
    different layers should share the model parameters.

    Args:
      `dims` is an integer list, in which two adjacent elements stand for the
      input and output dimensions of the corresponding layer. The length of
      `dims` is not less than 2. If len(`dims`) > 2, hidden layers will be
      constructed.

     `num_head` is an integer or integer list, means number of head for each layer.

      e.g.
      `dims = [128, 256, 256, 64]`, means the input dimension is 128 and the
      output dimension is 64. Meanwhile, 2 hidden layers with dimension 256 will
      be added between the input and output layer.
    """
    assert len(dims) > 2
    if isinstance(num_head, int):
      num_head = [num_head]

    layers = []
    for i in range(len(dims) - 1):
      layer = EgoGATLayer("homo_" + str(i),
                          input_dim=dims[i],
                          output_dim=dims[i + 1],
                          num_head=num_head[i % len(num_head)],
                          attn_dropout=attn_dropout)
      # If the len(dims) = K, it means that (K-1) LEVEL layers will be added. At
      # each LEVEL, computation will be performed for each two adjacent hops,
      # such as (nodes, hop1), (hop1, hop2) ... . We have (K-1-i) such pairs at
      # LEVEL i. In a homogeneous graph, they will share model parameters.
      group = EgoGATLayerGroup([layer] * (len(dims) - 1 - i))
      layers.append(group)

    super(HomoEgoGAT, self).__init__(layers, bn_fn, active_fn, dropout)

