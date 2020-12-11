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
"""Class of GraphSage Convolutional layer"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

from graphlearn.python.model.tf.aggregators import GCNAggregator
from graphlearn.python.model.tf.aggregators.mean_aggregator import MeanAggregator
from graphlearn.python.model.tf.aggregators import SumAggregator
from graphlearn.python.model.tf.layers.conv import BaseConv


class GraphSageConv(BaseConv):
  """GraphSage Convolutional layer.

  The GraphSAGE layer takes batch node's embedding and corresponding neighbors
  embedding as inputs, then aggregates embedding for each node of the batch.

  Args:
    in_dim: Input dimension.
    out_dim: Output dimension.
    agg_type: Aggregation type, can be 'gcn', 'mean', 'sum'
    act: Activation function.
    name: User defined name.
  """

  def __init__(self, index, in_dim, out_dim, agg_type, act=tf.nn.relu, name=''):
    self._in_dim = in_dim
    self._out_dim = out_dim

    if agg_type == 'gcn':
      self._aggregator = GCNAggregator(index,
                                       input_dim=in_dim,
                                       output_dim=out_dim,
                                       act=act,
                                       name=name + "agg_%d" % (index))
    elif agg_type == 'mean':
      self._aggregator = MeanAggregator(index,
                                        input_dim=in_dim,
                                        output_dim=out_dim,
                                        act=act,
                                        name=name + "agg_%d" % (index))
    else:
      self._aggregator = SumAggregator(index,
                                       input_dim=in_dim,
                                       output_dim=out_dim,
                                       act=act,
                                       name=name + "agg_%d" % (index))

  def forward(self, self_vecs, neigh_vecs):
    """
    Neighbors' embeddings are aggregated to update node's self embedding

    Args:
      self_vecs: batch nodes' embedding vector, shape [B, D]
      neigh_vecs: corresponding neighbor nodes' embedding vector, shape [total_nbrs, D]

    Returns:
      aggregated nodes' embedding vector [B, H]
    """
    return self._aggregator.aggregate(self_vecs=self_vecs,
                                      neigh_vecs=neigh_vecs)
