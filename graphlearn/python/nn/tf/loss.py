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
"""Loss functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Unsupervised loss.
def sigmoid_cross_entropy_loss(pos_logit,
                               neg_logit):
  """Sigmoid cross entropy loss.
  Args:
    pos_logit: positive logits, tensor with shape [batch_size]
    neg_logit: negative logits, tensor with shape [batch_size]
  Returns:
    loss, a scalar tensor
  """
  true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(pos_logit), logits=pos_logit)
  negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.zeros_like(neg_logit), logits=neg_logit)
  loss = tf.reduce_mean(true_xent) + 1.0 * tf.reduce_mean(negative_xent)
  return loss

def unsupervised_softmax_cross_entropy_loss(src_emb,
                                            pos_emb,
                                            neg_emb,
                                            temperature=1.0):
  """Softmax cross entropy loss.
  This loss is mostly used for inner product based two-tower model 
  for recommdener systems.
  Args:
    src_emb: src embedding, tensor with shape [batch_size, dim]
    pos_emb: positive dst embedding, tensor with shape [batch_size, dim]
    neg_emb: negative dst embedding, tensor with shape [batch_size * neg_num, dim]
  Returns:
    loss, a scalar tensor
  """
  pos_sim = tf.reduce_sum(tf.multiply(src_emb, pos_emb), axis=-1, keepdims=True)
  neg_sim = tf.matmul(src_emb, tf.transpose(neg_emb))

  logit = tf.nn.softmax(tf.concat([pos_sim, neg_sim] , axis=-1) / temperature)
  loss = -tf.reduce_mean(tf.log(logit[:, :1] + 1e-12))

  return loss


def triplet_margin_loss(pos_src_emb, pos_edge_emb, pos_dst_emb,
                        neg_src_emb, neg_edge_emb, neg_dst_emb,
                        margin, neg_num, L=1):
  """triplet margin loss for TransE.
  Args:
    pos_src_emb: positive src embedding, tensor with shape [batch_size, dim]
    pos_edge_emb: positive edge embedding, tensor with shape [batch_size, dim]
    pos_dst_emb: positive dst embedding, tensor with shape [batch_size, dim]
    neg_src_emb: negative src embedding, tensor with shape [batch_size, dim]
    neg_edge_emb: negative edge embedding, tensor with shape [batch_size, dim]
    neg_dst_emb: negative dst embedding, tensor with shape [batch_size, dim]
  Returns:
    loss, a scalar tensor
  """
  if L==2:
    pos_d = tf.reduce_sum(tf.square(pos_src_emb + pos_edge_emb - pos_dst_emb), axis=-1)
    neg_d = tf.reduce_sum(tf.square(neg_src_emb + neg_edge_emb - neg_dst_emb), axis=-1)
  else:
    pos_d = tf.reduce_sum(tf.abs(pos_src_emb + pos_edge_emb - pos_dst_emb), axis=-1)
    neg_d = tf.reduce_sum(tf.abs(neg_src_emb + neg_edge_emb - neg_dst_emb), axis=-1)
  if neg_num > 1:
    pos_d = tf.reshape(tf.tile(tf.expand_dims(pos_d, -1), [1, neg_num]), [-1])
  loss = tf.reduce_mean(tf.maximum(0.0, margin + pos_d - neg_d)) 
  return loss


def triplet_softplus_loss(pos_src_emb, pos_edge_emb, pos_dst_emb,
                          neg_src_emb, neg_edge_emb, neg_dst_emb):
  """triplet softplus loss for DistMult.
  Args:
    pos_src_emb: positive src embedding, tensor with shape [batch_size, dim]
    pos_edge_emb: positive edge embedding, tensor with shape [batch_size, dim]
    pos_dst_emb: positive dst embedding, tensor with shape [batch_size, dim]
    neg_src_emb: negative src embedding, tensor with shape [batch_size, dim]
    neg_edge_emb: negative edge embedding, tensor with shape [batch_size, dim]
    neg_dst_emb: negative dst embedding, tensor with shape [batch_size, dim]
  Returns:
    loss, a scalar tensor
  """
  pos_s = tf.reduce_sum(pos_src_emb * pos_edge_emb * pos_dst_emb, axis=-1)
  neg_s = tf.reduce_sum(neg_src_emb * neg_edge_emb * neg_dst_emb, axis=-1)
  loss = (tf.reduce_mean(tf.nn.softplus(-pos_s)) +
          tf.reduce_mean(tf.nn.softplus(neg_s))) / 2
  return loss