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
"""GCN convolution layer"""
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


class GCNConv(BaseConv):
  """Base GCN convolutional layer.

  Args:
    out_dim: Output dimension.
    act: Activation function.
    name: User defined name.
  """
  def __init__(self, out_dim, act=tf.nn.relu, name=''):
    self._fc = tf.keras.layers.Dense(units=out_dim,
                                     activation=act,
                                     use_bias=False,
                                     name=name+'w')

  def forward(self, self_vecs, neigh_vecs, segment_ids=None):
    """ Update node's embedding based on its neighbors.

    Args:
      self_vecs: batch nodes' embeddings with shape [B, D]
      neigh_vecs: neighbor nodes' embeddings with shape [total_nbrs, D]
      segment_ids: segment ids that indicates neighbor nodes' belonging,
      shape [total_nbrs]

    Returns:
      updated batch nodes' embedding vector [B, H]
    """
    if segment_ids is None:  # sampled GCN
      neigh_vecs = tf.reduce_sum(neigh_vecs, axis=1)
    else:  # full neighbor GCN
      neigh_vecs = tf.segment_sum(data=neigh_vecs, segment_ids=segment_ids)
    updated_vecs = tf.reduce_sum([self_vecs, neigh_vecs], axis=0)
    return self._fc(updated_vecs)
