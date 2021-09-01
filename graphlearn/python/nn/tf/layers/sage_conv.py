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
"""SubGraph based GraphSAGE convolutional layer"""

import tensorflow as tf

from graphlearn.python.nn.tf.layers.sub_conv import SubConv


class SAGEConv(SubConv):
  def __init__(self, in_dim, out_dim,
               agg_type='mean',
               normalize=False,
               use_bias=False,
               name=''):
    self._in_dim = in_dim
    self._out_dim = out_dim
    self._agg_type = agg_type
    self._normalize = normalize
    self._bias = use_bias
    self._name = name

    self._vars = {}
    with tf.variable_scope(self._name + '/'  + 'layer',
                           reuse=tf.AUTO_REUSE):
      self._vars['self_weights'] = \
        tf.get_variable(shape=[self._in_dim, self._out_dim],
                        name='self_weights')
      self._vars['neigh_weights'] = \
        tf.get_variable(shape=[self._in_dim, self._out_dim],
                        name='neigh_weights')
      if self._bias:
        self._vars['bias'] =\
          tf.Variable(tf.zeros([self._out_dim], dtype=tf.float32), name='bias')

  def forward(self, edge_index, node_vec, **kwargs):
    """
    Args:
      edge_index: A Tensor. Edge index of subgraph.
      node_vec: A Tensor. node feature embeddings with shape
      [batchsize, dim].
    Returns:
      A tensor. output embedding with shape [batch_size, output_dim].
    """
    # aggregate
    nbr_input = tf.gather(node_vec, edge_index[1])
    if self._agg_type == 'sum' or self._agg_type == 'gcn':
      nbr_reduce_msg = tf.math.unsorted_segment_sum(nbr_input,
                                                    edge_index[0],
                                                    tf.shape(node_vec)[0])
    elif self._agg_type == 'mean':
      nbr_reduce_msg = tf.math.unsorted_segment_mean(nbr_input,
                                                     edge_index[0],
                                                     tf.shape(node_vec)[0])
    else:
      raise NotImplementedError('Unsupported agg_type.')

    # update
    from_neighs = tf.matmul(nbr_reduce_msg, self._vars['neigh_weights'])
    from_self = tf.matmul(node_vec, self._vars["self_weights"])

    if self._agg_type == 'gcn':
      out = from_neighs
    else:
      out = tf.add_n([from_self, from_neighs])

    if self._bias:
      out += self._vars['bias']
    if self._normalize:
      out = tf.nn.l2_normalize(out, 1)
    return out
