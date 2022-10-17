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

from graphlearn.python.nn.tf.config import conf
from graphlearn.python.nn.tf.module import Module
from graphlearn.python.nn.tf.layers.linear_layer import LinearLayer

class LinkPredictor(Module):
  """ link predictor.

  Args:
    name: The name of link predictor.
    input_dim: The Input dimension.
    num_layers: Number of hidden layers.
    active_fn: Activation function for hidden layers' output.
    dropout: Dropout rate for hidden layers' output. Default is 0.
  """

  def __init__(self,
               name,
               input_dim,
               num_layers,
               dropout=0.0):
    self.dropout = dropout
    self.layers = []

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      for i in range(num_layers):
        output_dim = 1 if i == num_layers - 1 else input_dim
        layer = LinearLayer("link_predictor_" + str(i),
                             input_dim=input_dim,
                             output_dim=output_dim,
                             use_bias=True)
        self.layers.append(layer)

  def forward(self, x):
    """
    Args:
      x: input Tensor of shape [batch_size, input_dim]
    Returns:
      logits: Output logits tensor with shape [batch_size]
    """
    for i in range(len(self.layers) - 1):
      x = self.layers[i](x)
      x = tf.nn.relu(x)
      if self.dropout and conf.training:
        x = tf.nn.dropout(x, keep_prob=1-self.dropout)
    # the output logits
    logits = tf.squeeze(self.layers[-1](x))
    return logits