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


class EgoRGCN(tfg.Module):
  def __init__(self,
               input_dim,
               hidden_dim,
               out_dim,
               num_layers,
               num_relations,
               num_bases=None,
               num_blocks=None,
               agg_type="mean",
               use_bias=False,
               bn_func=None,
               act_func=tf.nn.relu,
               dropout=0.0,
               **kwargs):
    """EgoGraph based RGCN. 
  
    Args:
      input_dim: input dimension of nodes.
      hidden_dim: hidden dimension of nodes.
      out_idm: output dimension of nodes.
      num_relations: Number of relations.
      num_bases: Denotes the number of bases to use, default is None.
      num_blocks: Denotes the number of blocks to use, default is None.
        Note that num_bases and num_blocks cannot be set at the same time.
      agg_type: A string, aggregation strategy. The optional values are
        'mean', 'sum', 'max'.
      use_bias: A boolean, whether add bias after computation.
      bn_func: Batch normalization function for hidden layers' output. Default is
        None, which means batch normalization will not be performed.
      act_func: Activation function for hidden layers' output. 
        Default is tf.nn.relu.
      dropout: Dropout rate for hidden layers' output. Default is 0.0, which
        means dropout will not be performed. The optional value is a float.
    """
    self.num_relations = num_relations
    self.bn_func = bn_func
    self.active_func = act_func
    self.dropout = dropout
    self.layers = []
    
    for i in range(num_layers):
      in_dim = input_dim if i == 0 else hidden_dim
      h_dim = out_dim if i == num_layers - 1 else hidden_dim
      conv = tfg.EgoRGCNConv("rgcn_" + str(i),
                              in_dim,
                              h_dim,
                              num_relations,
                              num_bases=num_bases,
                              num_blocks=num_blocks,
                              agg_type=agg_type,
                              use_bias=use_bias)
      self.layers.append(conv)

  def forward(self, x_list, expands):
    """ Update node embeddings.
    Args:
      x_list: A list of list, representing input nodes and their K-hop neighbors.
        The first element x_list[0] is a list with one element which means root node tensor 
        with shape`[n, input_dim]`.
        The following element x_list[i] (i > 0) is i-th hop neighbors list with legth num_relations^i.
        It consists of different types of neighbors, and each element of x_list[i] is a tensor with 
        shape `[n * k_1 * ... * k_i, input_dim]`, where `k_i` means the neighbor count of each node 
        at i-th hop of root node. 

        Note that the elements of each list must be stored in the same order when stored 
        by relation type. For example, their are 2 relations and 2-hop neighbors, the x_list is stored
        in the following format:
        [[root_node], 
         [hop1_r1, hop1_r2], 
         [hop1_r1_hop2_r1, hop1_r1_hop2_r2, hop1_r2_hop2_r1, hop1_r2_hop2_r2]]

      expands: An integer list of neighbor count at each hop. For the above
        x_list, expands = [k_1, k_2, ... , k_K]

    Returns:
      A tensor with shape `[n, output_dim]`.
    """
    depth = len(expands)
    assert depth == (len(x_list) - 1)
    assert depth == len(self.layers)

    H = x_list
    for layer_idx in range(len(self.layers)): # for each conv layers.
      tmp_vecs = []
      num_root = 1 # the number of root node at each hop.
      for hop in range(depth - layer_idx): # for each hop neighbor, h[i+1]->h[i]
        tmp_nbr_vecs = []
        for offset in range(num_root): # do h[i+1]->h[i] according different relations.
          src_vecs = H[hop][offset]
          neigh_vecs = H[hop+1][(offset*self.num_relations) : ((offset+1)*self.num_relations)]
          h = self.layers[layer_idx].forward(src_vecs, neigh_vecs, expands[hop])
          if self.bn_func is not None:
            h = self.bn_func(h)
          if self.active_func is not None:
            h = self.active_func(h)
          if self.dropout and tfg.conf.training:
            h = tf.nn.dropout(h, keep_prob=1-self.dropout)
          tmp_nbr_vecs.append(h)
        num_root *= self.num_relations # the root node of the next hop is expand by num_relations.
        tmp_vecs.append(tmp_nbr_vecs)
      H = tmp_vecs

    return H[0][0]