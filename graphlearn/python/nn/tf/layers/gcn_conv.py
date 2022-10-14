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
"""SubGraph based GCN convolutional layer"""

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

from graphlearn.python.nn.tf.layers.sub_conv import SubConv


class GCNConv(SubConv):
  def __init__(self, in_dim, out_dim,
               normalize=True,
               use_bias=False,
               name=''):
    self._in_dim = in_dim
    self._out_dim = out_dim
    self._normalize = normalize
    self._bias = use_bias
    self._name = name

    self._vars = {}
    with tf.variable_scope(self._name + '/'  + 'layer',
                           reuse=tf.AUTO_REUSE):
      self._vars['weights'] = \
        tf.get_variable(shape=[self._in_dim, self._out_dim],
                        name='weights')
      if self._bias:
        self._vars['bias'] =\
          tf.Variable(tf.zeros([self._out_dim], dtype=tf.float32), name='bias')

  def gcn_norm(self, edge_index, num_nodes):
    # add self-loop.
    diagnal_edge_index = tf.stack([tf.range(num_nodes, dtype=tf.int32)] * 2, axis=0)
    edge_index = tf.concat([edge_index, diagnal_edge_index], axis=1)
    edge_weight = tf.ones(tf.shape(edge_index)[1], dtype=tf.float32)
    deg = tf.unsorted_segment_sum(edge_weight, edge_index[0], num_nodes)
    deg_inv_sqrt = tf.pow(deg, -0.5)
    return edge_index, deg_inv_sqrt

  def forward(self, edge_index, node_vec, **kwargs):
    """
    Args:
      edge_index: A Tensor. Edge index of subgraph.
      node_vec: A Tensor. node feature embeddings with shape
      [batchsize, dim].
    Returns:
      A tensor. output embedding with shape [batch_size, output_dim].
    """
    num_nodes = tf.shape(node_vec)[0]
    updated_vec = tf.matmul(node_vec, self._vars['weights'])
    if self._normalize:
      edge_index, edge_norm = self.gcn_norm(edge_index, num_nodes)
      # GCN degree norm.
      updated_vec = updated_vec * tf.expand_dims(edge_norm, 1)
    nbr_input = tf.gather(updated_vec, edge_index[1])
    out = tf.math.unsorted_segment_sum(nbr_input,
                                       edge_index[0],
                                       num_nodes)
    if self._bias:
      out += self._vars['bias']
    return out