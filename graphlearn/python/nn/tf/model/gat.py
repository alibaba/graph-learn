# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
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
'''SubGraph based GAT.'''

import tensorflow as tf

from graphlearn.python.nn.tf.config import conf
from graphlearn.python.nn.tf.module import Module
from graphlearn.python.nn.tf.layers.gat_conv import GATConv


class GAT(Module):
  def __init__(self,
               batch_size,
               hidden_dim,
               output_dim,
               depth=2,
               drop_rate=0.0,
               attn_heads=1,
               attn_drop=0.0,
               **kwargs):
    self.depth = depth
    self.drop_rate = drop_rate
    self.attn_drop = attn_drop

    self.layers = []
    for i in range(depth):
      output_dim = output_dim if i == depth - 1 else hidden_dim
      num_heads = 1 if (i == depth - 1 and depth != 1) else attn_heads
      concat = True if (i == depth - 1 and depth != 1) else False
      self.layers.append(GATConv(output_dim,
                                 num_heads=num_heads,
                                 concat=concat,
                                 dropout=self.attn_drop,
                                 name='conv' + str(i)))

  def forward(self, batchgraph):
    h = batchgraph.transform().nodes
    for l, layer in enumerate(self.layers):
      h = layer.forward(batchgraph.edge_index, h)
      if l != self.depth - 1:
        h = tf.nn.relu(h)
        if self.drop_rate and conf.training:
          h = tf.nn.dropout(h, 1 - self.drop_rate)
    src = tf.gather(h, batchgraph.graph_node_offsets)
    dst = tf.gather(h, batchgraph.graph_node_offsets + 1)
    return src, dst