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
from graphlearn.python.nn.tf.config import conf
from graphlearn.python.nn.tf.module import Module
from graphlearn.python.nn.tf.layers.linear_layer import LinearLayer


class NodeClassifier(Module):
  """ Node classifier.

  Args:
    `dims` is an integer list, in which two adjacent elements stand for the
    input and output dimensions of the corresponding layer. We will add the
    output layer with dimension `class_num` automatically.

    e.g.
    `dims = [256, 64, 16]`, means 3 layers with shape (256, 64), (64, 16)
    and (16, 2) exist. The classifier will take inputs whose dimension must
    be 256.

    class_num: Default is 2, binary classifier.
    active_fn: Activation function for hidden layers' output.
    dropout: Dropout rate for hidden layers' output. Default is None, which
      means dropout will not be performed. The optional value is a float.
  """

  def __init__(self,
               dims,
               class_num=2,
               active_fn=tf.nn.relu,
               dropout=None):
    self.class_num = class_num
    self.active_func = active_fn

    dims.append(class_num)
    self.layers = []
    for i in range(len(dims) - 1):
      layer = LinearLayer("node_classifier_" + str(i),
                           input_dim=dims[i],
                           output_dim=dims[i + 1],
                           use_bias=True)
      self.layers.append(layer)

    if dropout is not None:
      self.dropout_func = lambda x: tf.nn.dropout(x, keep_prob=1-dropout)
    else:
      self.dropout_func = None

  def forward(self, x, y):
    """ Return the probabilities based on which `x` belongs to the corresponding
    class, as well as the final loss compared with the ground truth `y`. The
    shape of the probabilities is [batch_size, class_num].

    x: The input tensor with shape [batch_size, dim], where dim must match the
      first layer.
    y: The label tensor with shape [batch_size]. The value of labels belongs to
      [0, class_num). For example, each label value must be 0 or 1 for binary
      classification.
    """
    for i in range(len(self.layers) - 1):
      x = self.layers[i].forward(x)
      if self.active_func:
        x = self.active_func(x)
      if self.dropout_func and conf.training:
        x = self.dropout_func(x)

    # the output logits
    logits = self.layers[-1].forward(x)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(loss)
    return logits, loss
