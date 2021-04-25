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
from graphlearn.python.nn.tf.data.feature_group import FeatureHandler
from graphlearn.python.nn.tf.layers.linear_layer import LinearLayer

class InputLayer(Module):
  """ Transform a vertex or an Edge into tensors. A wrapper of FeatureHandler.

  Args:
    name: A string, layer name.
    feature_spec: A FeatureSpec object to describe the coming vertices or edges.
    output_dim: An integer, if not None, a LinearLayer will be applied after
      FeatureHandler's output.
    use_bias: A boolean, add bias or not. It is used just when the output_dim
      is not None.
  """

  def __init__(self,
               name,
               feature_spec,
               output_dim=None,
               use_bias=False,
               **kwargs):
    super(InputLayer, self).__init__()

    with tf.variable_scope("input_layer", reuse=tf.AUTO_REUSE):
      self.handler = FeatureHandler(name, feature_spec)
      self.linear = LinearLayer(name,
                                feature_spec.dimension,
                                output_dim,
                                use_bias) if output_dim is not None else None

  def forward(self, x):
    output = self.handler(x)
    return output if self.linear is None else self.linear(output)
