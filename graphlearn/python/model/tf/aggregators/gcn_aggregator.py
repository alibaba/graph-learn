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
"""Class for gcn aggregator"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

from graphlearn.python.model.tf.aggregators import BaseAggregator
from graphlearn.python.model.tf.utils.inits import zeros

DNN_PT_SIZE = 32 * 1024


class GCNAggregator(BaseAggregator):
  """class for GCN aggregator.

  Args:
    index: Index for the convolution layer this aggregator belongs to.
    input_dim: Input dimension.
    output_dim: Output dimension.
    neigh_input_dim: Neighbor nodes' input dimension.
    ps_num: Parameter worker count in distributed model.
    bias: Bool, set to True if use bias.
    act: Activation function.
    name: User defined name.
  """


  def __init__(self,
               index,
               input_dim,
               output_dim=None,
               neigh_input_dim=None,
               ps_num=0,
               bias=False,
               act=tf.nn.relu,
               name=''):
    self._index = index
    self._name = name
    self._bias = bias
    self._act = act
    self._vars = {}

    self._input_dim = input_dim
    self._neigh_input_dim = neigh_input_dim
    self._output_dim = output_dim
    if self._neigh_input_dim is None:
      self._neigh_input_dim = input_dim
    if self._output_dim is None:
      self._output_dim = input_dim

    self._partitioner = None
    if ps_num:
      self._partitioner = tf.min_max_variable_partitioner(
          max_partitions=ps_num,
          min_slice_size=DNN_PT_SIZE)

  def aggregate(self, self_vecs, neigh_vecs):
    with tf.variable_scope(self._name + '/' + str(self._index) + '_layer',
                           reuse=tf.AUTO_REUSE,
                           partitioner=self._partitioner):
      self._vars['weights'] = \
        tf.get_variable(shape=[self._neigh_input_dim, self._output_dim],
                        name='weights')
      if self._bias:
        self._vars['bias'] = zeros([self._output_dim], name='bias')

      means = tf.reduce_mean(tf.concat(
          [neigh_vecs, tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)

      # [nodes] x [out_dim]
      output = tf.matmul(means, self._vars['weights'])
      # bias
      if self._bias:
        output += self._vars['bias']
      output = tf.reshape(output, shape=[-1, self._output_dim])
    if self._act is not None:
      return self._act(output)
    return output
