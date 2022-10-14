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

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

import graphlearn.python.nn.tf as tfg


class EgoGraphSAGE(tfg.EgoGNN):
  def __init__(self,
               dims,
               agg_type="mean",
               bn_func=None,
               act_func=tf.nn.relu,
               dropout=0.0,
               **kwargs):
    """ EgoGraph based GraphSAGE.

    Args:
      dims: An integer list, in which two adjacent elements stand for the
        input and output dimensions of the corresponding EgoLayer. The length
        of `dims` is not less than 2. If len(`dims`) > 2, hidden layers will 
        be constructed. 
        e.g. `dims = [128, 256, 256, 64]`, means the input dimension is 128 and
        the output dimension is 64. Meanwhile, 2 hidden layers with dimension 
        256 will be added between the input and output layer.
      agg_type: A string, aggregation strategy. The optional values are
        'mean', 'sum', 'max' and 'gcn'.
    """
    assert len(dims) > 1

    layers = []
    for i in range(len(dims) - 1):
      conv = tfg.EgoSAGEConv("homo_" + str(i),
                             in_dim=dims[i],
                             out_dim=dims[i + 1],
                             agg_type=agg_type)
      # If the len(dims) = K, it means that (K-1) LEVEL layers will be added. At
      # each LEVEL, computation will be performed for each two adjacent hops,
      # such as (nodes, hop1), (hop1, hop2) ... . We have (K-1-i) such pairs at
      # LEVEL i. In a homogeneous graph, they will share model parameters.
      layer = tfg.EgoLayer([conv] * (len(dims) - 1 - i))
      layers.append(layer)

    super(EgoGraphSAGE, self).__init__(layers, bn_func, act_func, dropout)

