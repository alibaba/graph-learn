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
"""Basic class of SubGraph based heterogeneous graph convolutional layer"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from collections import defaultdict

from graphlearn.python.nn.tf.module import Module


class HeteroConv(Module):
  """ Handles heterogeneous subgraph(`HeteroSubGraph`) convolution.
  
  This layer will perform the convolution operation according to the 
  specified edge type and its corresponding convolutional layer(`conv_dict`). 
  If multiple edges point to the same destination node, their results 
  will be aggregated according to `agg_type`.

  Args:
    conv_dict: A dict containing `SubConv` layer for each edge type.
    agg_type: The aggregation type used to specify the result aggregation 
      method when the same destination node has multiple edges.
      The optional values are: `sum`, `mean`, `min`, `max`, the default 
      value is `mean`. 
  """
  def __init__(self, conv_dict, agg_type='mean'):
    super(HeteroConv, self).__init__()
    self.conv_dict = conv_dict
    self.agg_type = agg_type

  def forward(self, edge_index_dict, node_vec_dict, **kwargs):
    """
    Args:
      edge_index_dict: A dict containing edge type to edge_index mappings.
      node_vec_dict: A dict containing node type to node_vec mappings.
    Returns:
      A dict containing node type to output embedding mappings.
    """
    out_dict = defaultdict(list)
    for edge_type, edge_index in edge_index_dict.items():
      h, r, t = edge_type
      if edge_type not in self.conv_dict:
        continue
      if h == t:
        out = self.conv_dict[edge_type](edge_index, node_vec_dict[h])
      else:
        out = self.conv_dict[edge_type](edge_index, [node_vec_dict[h], node_vec_dict[t]])
      out_dict[t].append(out)
    
    for k, v in out_dict.items():
      if len(v) == 1:
        out_dict[k] = v[0]
      else:
        out_dict[k] = getattr(tf.math, 'reduce_' + self.agg_type)(v, 0)
    
    return out_dict