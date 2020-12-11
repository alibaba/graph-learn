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
"""GAT convolution layer"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

from graphlearn.python.model.tf.layers.conv import BaseConv


class GATConv(BaseConv):
  """Base GAT convolutional layer.

  The GAT layer is mainly used to calculate attention coefficients
  for each neighbor node then aggregator neighbors' embeddings.

  Args:
    out_dim: Output dimension.
    attn_drop: Dropout ratio for attention coefficients.
    act: Activation function.
    name: User defined name.
  """

  def __init__(self, out_dim, attn_drop, act=tf.nn.relu, name=''):
    self._fc = tf.keras.layers.Dense(units=out_dim,
                                     use_bias=False,
                                     name=name + 'w')
    self._attn_fc = tf.keras.layers.Dense(units=1,
                                          use_bias=False,
                                          name=name + 'a')
    self._attn_drop = attn_drop
    self._activation = act

  def forward(self, self_vecs, neigh_vecs, segment_ids=None):
    """Calculates attention coefficients.

    The following code is used to implement equation (1) (2) (3) and (4)
    in the paper, which is called **self-attention** process.

    \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})
    e_{ij}^{l}  = \mathrm{LeakyReLU}(\vec{a}^T [W h_{i} \vert W h_{j}])

    Args:
      self_vecs: Tensor, batch nodes' embedding vector, shape [B, D]
      neigh_vecs: Tensor, corresponding neighbor nodes' embedding vector,
      shape [total_nbrs, D]
      segment_ids: Tensor, segment ids indicates neighbor nodes' belonging,
      shape [total_nbrs]

    Returns:
      updated neighbor nodes' embedding vector.
    """

    self_vecs = self._fc(self_vecs)
    neigh_vecs = self._fc(neigh_vecs)
    if segment_ids is None:  # sampled GAT
      num_neibors = neigh_vecs.shape[1]
      self_vecs_extend = tf.tile(tf.expand_dims(self_vecs, 1),
                                 [1, num_neibors, 1])
      coefficients = tf.nn.softmax(tf.nn.leaky_relu(self._attn_fc(
          tf.concat([self_vecs_extend, neigh_vecs], axis=-1))))
      coefficients = tf.nn.dropout(coefficients, 1 - self._attn_drop)
      neigh_vecs = tf.multiply(coefficients, neigh_vecs)
      neigh_vecs = tf.reduce_sum(neigh_vecs, axis=1)
    else:  # full neighbor GAT
      self_vecs_extend = tf.gather(self_vecs, segment_ids)
      coefficients = tf.math.exp(tf.nn.leaky_relu(self._attn_fc(
          tf.concat([self_vecs_extend, neigh_vecs], axis=-1))))
      seg_sum = tf.gather(tf.segment_sum(coefficients, segment_ids), segment_ids)
      coefficients = coefficients / seg_sum
      coefficients = tf.nn.dropout(coefficients, 1 - self._attn_drop)
      neigh_vecs = tf.multiply(coefficients, neigh_vecs)
      neigh_vecs = tf.segment_sum(neigh_vecs, segment_ids)

    if self._activation is not None:
      neigh_vecs = self._activation(neigh_vecs)

    return neigh_vecs


class MultiHeadGATConv(object):
  """class for mulit head GAT convolutional layer

  Args:
    out_dim: Output dimension.
    num_head: number of heads.
    attn_drop: Dropout ratio for attention coefficients.
    concat: Bool, set to True if concatenate embeddings from all heads
    act: Activation coefficients.
  """

  def __init__(self,
               out_dim,
               num_head,
               attn_drop=0.0,
               concat=False,
               act=tf.nn.relu):
    self.gat_convs = [GATConv(out_dim, attn_drop, act)
                      for _ in range(num_head)]
    self._concat = concat

  def forward(self, self_vecs, neigh_vecs, sigment_ids=None):
    """ To stabilize the learning process of self-attention,
    authors propose to use multiple attetnion mechanism,
    the following code is used to implement equation (5) and (6) in the paper.

    Args:
      self_vecs: batch nodes' embedding vector, shape [B, D]
      neigh_vecs: corresponding neighbor nodes' embedding
      with shape [total_nbrs, D]
      segment_ids: segment ids indicates neighbor nodes' belonging,
      shape [total_nbrs]

    Returns:
      updated batch nodes' embedding vector
    """
    outputs = [conv.forward(self_vecs, neigh_vecs, sigment_ids)
               for conv in self.gat_convs]
    if self._concat:
      return tf.concat(outputs, axis=-1)
    return tf.reduce_mean(outputs, axis=0)
