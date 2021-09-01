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
"""SubGraph based GAT convolutional layer"""

import tensorflow as tf

from graphlearn.python.nn.tf.config import conf
from graphlearn.python.nn.tf.layers.sub_conv import SubConv
from graphlearn.python.nn.tf.utils.softmax import unsorted_segment_softmax


class GATConv(SubConv):
  """multi-head GAT convolutional layer.
  """
  def __init__(self,
               out_dim,
               num_heads=1,
               concat=False,
               dropout=0.0,
               use_bias=False,
               name=''):
    self._out_dim = out_dim
    self._num_heads = num_heads
    self._concat = concat
    self._dropout =  dropout
    self._bias = use_bias
    self._name = name

    self._vars = {}
    with tf.variable_scope(self._name + '/'  + 'layer',
                           reuse=tf.AUTO_REUSE):
      self._vars['attn_src'] = \
        tf.get_variable(shape=[1, self._num_heads, self._out_dim],
                        name='attention_weights_src')
      self._vars['attn_dst'] = \
        tf.get_variable(shape=[1, self._num_heads, self._out_dim],
                        name='attention_weights_dst')
      self._linear = \
          tf.keras.layers.Dense(units=self._num_heads*self._out_dim,
                                use_bias=False,
                                name=name + 'w')
      if self._bias:
        if self._concat:
          self._vars['bias'] =\
            tf.Variable(tf.zeros([self._out_dim], dtype=tf.float32),
                        name='bias')
        else:
          self._vars['bias'] =\
            tf.Variable(tf.zeros([self._num_heads * self._out_dim], dtype=tf.float32),
                        name='bias')

  def forward(self, edge_index, node_vec, **kwargs):
    """
    Multi-head attention coefficients are computed as following:
      1.compute W*h_i(node_vec) using self._linear(node_vec).
      2.compute src_e and dst_e individually using 'attn_src' and 'attn_dst'
      3.add src_e and dst_e as e_ij and apply LeakyReLU.
      4.compute alpha_ij using unsorted_segment_softmax.

    Args:
      edge_index: A Tensor. Edge index of subgraph.
      node_vec: A Tensor. Node feature embeddings with shape
      [batch_size, dim].
    Returns:
      A Tensor. Output embedding with shape [batch_size, output_dim].
    """
    num_nodes = tf.shape(node_vec)[0]
    # add self-loop.
    diagnal_edge_index = tf.stack([tf.range(num_nodes, dtype=tf.int32)] * 2, axis=0)
    edge_index = tf.concat([edge_index, diagnal_edge_index], axis=1)
    # [batch_size, num_heads, output_dim]
    src_h = dst_h = tf.reshape(self._linear(node_vec),
                               [-1, self._num_heads, self._out_dim])

    # When computing attention coefficients alpha, the GAT concatenates src_h
    # and dst_h at first. Here we first compute src_e and dst_e individually,
    # and then add them as final coefficients in order to reduce memory usage.
    # [batch_size, num_heads]
    src_e = tf.reduce_sum(src_h * self._vars['attn_src'], axis=-1)
    dst_e = tf.reduce_sum(dst_h * self._vars['attn_dst'], axis=-1)
    # [num_edges, num_heads]
    src_e = tf.gather(src_e, edge_index[0])
    dst_e = tf.gather(dst_e, edge_index[1])
    e = tf.nn.leaky_relu(src_e + dst_e)
    alpha = unsorted_segment_softmax(e, edge_index[0], num_nodes)
    if self._dropout and conf.training:
      alpha = tf.nn.dropout(alpha, 1 - self._dropout)
    # [num_edges, num_heads, output_dim]
    nbr_input = tf.gather(dst_h, edge_index[1])
    alpha = tf.tile(tf.expand_dims(alpha, axis=2),
                    [1, 1, self._out_dim])
    # [batch_size, num_heads, output_dim]
    out = tf.math.unsorted_segment_sum(nbr_input * alpha,
                                       edge_index[0],
                                       num_nodes)
    if self._concat:
      out = tf.concat(tf.split(out, self._num_heads, axis=1), axis=1)
    else:
      out = tf.reduce_mean(out, axis=1)
    if self._bias:
      out += self._vars['bias']
    return out
