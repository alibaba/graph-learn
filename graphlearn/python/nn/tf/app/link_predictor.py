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

class LinkPredictor(Module):
  """ link predictor.

  Args:
    `dims` is an integer list, in which two adjacent elements stand for the
    input and output dimensions of the corresponding layer. We will add the
    output layer with dimension 1 automatically.

    e.g.
    `dims = [256, 64, 16]`, means 3 layers with shape (256, 64), (64, 16)
    and (16, 1) exist. The classifier will take inputs whose dimension must
    be 256.

    active_fn: Activation function for hidden layers' output.
    dropout: Dropout rate for hidden layers' output. Default is None, which
      means dropout will not be performed. The optional value is a float.
  """

  def __init__(self,
               name,
               dims,
               active_fn=tf.nn.relu,
               dropout=None):
    self.active_func = active_fn
    dims.append(1)
    self.layers = []

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      for i in range(len(dims) - 1):
        layer = LinearLayer("link_predictor_" + str(i),
                             input_dim=dims[i],
                             output_dim=dims[i + 1],
                             use_bias=True)
        self.layers.append(layer)

    if dropout is not None:
      self.dropout_func = lambda x: tf.nn.dropout(x, keep_prob=1-dropout)
    else:
      self.dropout_func = None

  def predict(self, x):
    for i in range(len(self.layers) - 1):
      x = self.layers[i].forward(x)
      if self.active_func:
        x = self.active_func(x)
      if self.dropout_func and conf.training:
        x = self.dropout_func(x)

    # the output logits
    logits = tf.squeeze(self.layers[-1].forward(x))
    return logits

class SupervisedLinkPredictor(LinkPredictor):
  def __init__(self,
               name,
               dims,
               active_fn=tf.nn.relu,
               dropout=None):
    super(SupervisedLinkPredictor, self).__init__(
        name, dims, active_fn, dropout)

  def forward(self, src, dst, labels):
    """ Return the similarity with shape [batch_size] between `src` and `dst`,
    as well as the final loss compared with the ground truth `labels`.

    src: The first input tensor with shape [batch_size, dim], where dim must
      match the first layer.
    dst: The second input tensor, whose shape must be the same with `src`.
    labels: A tensor with shape [batch_size], each value must be 1 or 0,
      indicating whether the link between `src` and `dst` exists or not.
    """
    # The more similar of src and dst, the larger of x
    x = 0 - tf.pow(src - dst, 2)
    logits = self.predict(x)
    labels = tf.cast(labels, logits.dtype)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return logits, tf.reduce_mean(loss)


class UnsupervisedLinkPredictor(LinkPredictor):
  def __init__(self,
               name,
               dims,
               active_fn=tf.nn.relu,
               dropout=None):
    super(UnsupervisedLinkPredictor, self).__init__(
        name, dims, active_fn, dropout)

  def forward(self, src, dst, neg):
    """ Return the final loss based on negative sampling.

    src: An input tensor with shape [batch_size, dim], where dim must match the
      first layer.
    dst: A tensor with the same shape of `src`. It has real links to `src`.
    neg: A tensor with shape [batch_size, neg_num, dim], which is generated by
      negative sampling and has no links to `src`.
    """
    x1 = src * dst
    true_logits = self.predict(x1)
    true_logits = tf.squeeze(true_logits)

    dim = src.shape[1]
    neg_expand = neg.shape[1]
    src = tf.tile(tf.expand_dims(src, axis=1), [1, neg_expand, 1])
    src = tf.reshape(src, [-1, dim])
    neg = tf.reshape(neg, [-1, dim])
    x2 = src * neg
    neg_logits = self.predict(x2)
    neg_logits = tf.squeeze(neg_logits)

    true_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(true_logits),
        logits=true_logits)
    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(neg_logits),
        logits=neg_logits)
    loss = tf.reduce_mean(true_loss) + tf.reduce_mean(neg_loss)
    return loss
