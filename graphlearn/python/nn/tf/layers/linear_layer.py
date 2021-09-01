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

import tensorflow as tf
from graphlearn.python.nn.tf.module import Module


class LinearLayer(Module):
  """ Do the computation `y = xW + B`.
    x: [-1, input_dim]
    W: [intput_dim, output_dim]
    B: [output_dim]
    y: [-1, output_dim]

  Args:
    name: A string, layer name.
    input_dim: An integer, the first dimension of W.
    output_dim: An integer, the second dimension of W.
    use_bias: A boolean, add bias or not.
  """

  def __init__(self,
               name,
               input_dim,
               output_dim,
               use_bias=False,
               **kwargs):
    super(LinearLayer, self).__init__()

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      self.weights = tf.get_variable(
          name="weights",
          shape=[input_dim, output_dim])

      if use_bias:
        self.bias = tf.get_variable(
            name="bias",
            shape=[output_dim])
      else:
        self.bias = None

  def forward(self, x):
    """ Return y = matmul(x, w) + b.
    """
    y = tf.matmul(x, self.weights)
    if self.bias is not None:
      y = tf.add(y, self.bias)
    return y
