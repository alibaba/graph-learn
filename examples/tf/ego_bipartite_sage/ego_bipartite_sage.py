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
import graphlearn.python.nn.tf as tfg


class EgoBipartiteGraphSAGE(tfg.EgoGNN):
  def __init__(self,
               src_input_dim,
               dst_input_dim,
               hidden_dims,
               agg_type="mean",
               bn_func=None,
               act_func=tf.nn.relu,
               dropout=0.0,
               **kwargs):
    """EgoGraph based Bipartite GraphSAGE. 
  
    Args:
      src_input_dim: input dimension of src nodes.
      dst_input_dim: input dimension of dst nodes.
      hidden_dims: An integer list, in which two adjacent elements stand for 
        the input and output dimensions of the corresponding EgoLayer.
      agg_type: A string, aggregation strategy. The optional values are
        'mean', 'sum', 'max'.
    """
    layers = []
    # input layer
    conv_src2dst = tfg.EgoSAGEConv("src_to_dst",
                                   in_dim=(src_input_dim, dst_input_dim),
                                   out_dim=hidden_dims[0],
                                   agg_type=agg_type)
    conv_dst2src = tfg.EgoSAGEConv("dst_to_src",
                                   in_dim=(dst_input_dim, src_input_dim),
                                   out_dim=hidden_dims[0],
                                   agg_type=agg_type)
    # the meta-path is 'src->dst->src->dst...'
    input_layer = tfg.EgoLayer([(conv_src2dst, conv_dst2src)[i % 2] for i in range(len(hidden_dims))])
    layers.append(input_layer)

    # hidden and output layer
    for i in range(len(hidden_dims)-1): # for each EgoLayer.
      conv = tfg.EgoSAGEConv("hidden_" + str(i),
                             in_dim=hidden_dims[i],
                             out_dim=hidden_dims[i + 1],
                             agg_type=agg_type)
      # If the len(dims) = K, it means that (K-1) LEVEL layers will be added. At
      # each LEVEL, computation will be performed for each two adjacent hops,
      # such as (nodes, hop1), (hop1, hop2) ... . We have (K-1-i) such pairs at
      # LEVEL i. In a homogeneous graph, they will share model parameters.
      layer = tfg.EgoLayer([conv] * (len(hidden_dims) - 1 - i))
      layers.append(layer)

    super(EgoBipartiteGraphSAGE, self).__init__(layers, bn_func, act_func, dropout)

