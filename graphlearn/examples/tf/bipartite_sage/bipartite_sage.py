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
'''SubGraph based BipartiteGraphSAGE.'''

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

from graphlearn.python.nn.tf.config import conf
from graphlearn.python.nn.tf.module import Module
from graphlearn.python.nn.tf.layers.hetero_conv import HeteroConv
from graphlearn.python.nn.tf.layers.sage_conv import SAGEConv

class BipartiteGraphSAGE(Module):
  def __init__(self,
               src_input_dim,
               dst_input_dim,
               hidden_dim,
               output_dim,
               depth=2,
               agg_type="mean",
               bn_func=None,
               act_func=tf.nn.relu,
               drop_rate=0.0,
               **kwargs):
    """EgoGraph based Bipartite GraphSAGE. 
  
    Args:
      src_input_dim: input dimension of src nodes.
      dst_input_dim: input dimension of dst nodes.
      agg_type: A string, aggregation strategy. The optional values are
        'mean', 'sum', 'max', 'gcn'.
    """
    self.depth = depth
    self.drop_rate = drop_rate

    self.layers = []
    for i in range(depth):
      ui_input_dim = [src_input_dim, dst_input_dim] if i == 0 else [hidden_dim, hidden_dim]
      iu_input_dim = [dst_input_dim, src_input_dim] if i == 0 else [hidden_dim, hidden_dim]
      output_dim = output_dim if i == depth - 1 else hidden_dim
      ui_conv = SAGEConv(ui_input_dim, output_dim,
                         agg_type=agg_type, name='ui_conv' + str(i))
      iu_conv = SAGEConv(iu_input_dim, output_dim,
                         agg_type=agg_type, name='iu_conv' + str(i))
      hetero_conv = HeteroConv({('u', 'u-i', 'i'): ui_conv,
                                ('i', 'u-i_reverse', 'u'): iu_conv})
      self.layers.append(hetero_conv)

  def forward(self, batchgraph):
    h_dict = batchgraph.transform().nodes_dict
    for l, layer in enumerate(self.layers):
      h_dict = layer.forward(batchgraph.edge_index_dict, h_dict)
      if l != self.depth - 1:
        h_dict = {key: tf.nn.relu(h) for key, h in h_dict.items()}
        if self.drop_rate and conf.training:
          h_dict = {key: tf.nn.dropout(h, 1 - self.drop_rate) for key, h in h_dict.items()}
    # return node embeddings of the original traversal of the u-i edge.
    h_u = tf.gather(h_dict['u'], batchgraph.graph_node_offsets_dict['u'])
    h_i = tf.gather(h_dict['i'], batchgraph.graph_node_offsets_dict['i'])
    return h_u, h_i