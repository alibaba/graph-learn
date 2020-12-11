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
"""Loss functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

#similarity measure function
def _rank_dot_product(vec1, vec2):
  return tf.reduce_sum(tf.multiply(vec1, vec2, name='rank_dot_prod'), axis=-1)

def _rank_euclidean_distance(vec1, vec2):
  return tf.sqrt(tf.reduce_sum(tf.square(vec1-vec2), -1), name='rank_euc_prod')

def _rank_cosine_distance(vec1, vec2):
  x1_norm = tf.sqrt(tf.reduce_sum(tf.square(vec1), axis=-1))
  x2_norm = tf.sqrt(tf.reduce_sum(tf.square(vec2), axis=-1))
  x1_x2 = tf.reduce_sum(tf.multiply(vec1, vec2), axis=-1) / (x1_norm * x2_norm)
  return x1_x2


#Supervised loss.
def softmax_cross_entropy_loss(emb, label):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=label, logits=emb)
  return tf.reduce_mean(loss)


# Unsupervised loss.
def sigmoid_cross_entropy_loss(src_emb,
                               pos_emb,
                               neg_emb,
                               sim_fn="dot"):
  """Sigmoid cross entropy loss for unsuperviesd model.
  Args:
    src_emb: tensor with shape [batch_size, dim]
    pos_emb: tensor with shape [batch_size, dim]
    neg_emb: tensor with shape [batch_size * neg_num, dim]
    sim_fn: similarity measure function
  Returns:
    loss, logit, label
  """
  emb_dim = tf.shape(src_emb)[1]
  batch_size = tf.shape(src_emb)[0]
  per_sample_neg_num = tf.shape(neg_emb)[0] / batch_size

  if sim_fn == "cosine":
    sim_function = _rank_cosine_distance
  elif sim_fn == "euclidean":
    sim_function = _rank_euclidean_distance
  elif sim_fn == "dot":
    sim_function = _rank_dot_product
  else:
    print("not support %s similarity measure function"%(sim_fn))
    raise Exception

  pos_logit = sim_function(src_emb, pos_emb)

  src_emb_exp = tf.tile(tf.expand_dims(src_emb, axis=1),
                        [1, per_sample_neg_num, 1])
  src_emb_exp = tf.reshape(src_emb_exp, [-1, emb_dim])
  neg_logit = sim_function(src_emb_exp, neg_emb)

  true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(pos_logit), logits=pos_logit)
  negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.zeros_like(neg_logit), logits=neg_logit)

  loss = tf.reduce_mean(true_xent) + 1.0 * tf.reduce_mean(negative_xent)
  logit = tf.concat([pos_logit, neg_logit], axis=-1)
  label = tf.concat([tf.ones_like(pos_logit, dtype=tf.int32),
                     tf.zeros_like(neg_logit, dtype=tf.int32)], axis=-1)

  return loss, logit, label


def sampled_softmax_loss(src_emb,
                         pos_ids,
                         neg_num,
                         output_emb_table,
                         output_emb_bias,
                         node_size,
                         s2h=True):
  """Sampled softmax loss.
  Args:
    src_emb: positive src embedding with shape [batch_size, dim]
    pos_ids: positive ids.
    output_emb_table:
    output_emb_bias:
    node_size: total node size.
    s2h: set True if need string to hash.
  """
  if s2h:
    pos_ids = tf.as_string(pos_ids)
    pos_ids = tf.string_to_hash_bucket_fast(
        pos_ids,
        node_size,
        name='softmax_loss_to_hash_bucket_oper')

  loss = tf.nn.sampled_softmax_loss(
      weights=output_emb_table,
      biases=output_emb_bias,
      labels=tf.reshape(pos_ids, [-1, 1]),
      inputs=src_emb,
      num_sampled=neg_num,
      num_classes=node_size,
      partition_strategy='mod',
      remove_accidental_hits=True)

  return [tf.reduce_mean(loss), None, None]


def kl_loss(src_emb, pos_emb, neg_emb, sim_fn="dot"):
  """kl loss used for line.
  Args:
    src_emb: tensor with shape [batch_size, dim]
    pos_emb: tensor with shape [batch_size, dim]
    neg_emb: tensor with shape [batch_size * neg_num, dim]
    sim_fn: similarity measure function, cosine, euclidean and dot.
  Returns:
    loss, logit, label
  """

  if sim_fn == "cosine":
    sim_function = _rank_cosine_distance
  elif sim_fn == "euclidean":
    sim_function = _rank_euclidean_distance
  elif sim_fn == "dot":
    sim_function = _rank_dot_product
  else:
    print("not support %s similarity measure function"%(sim_fn))
    raise Exception

  emb_dim = tf.shape(src_emb)[1]
  batch_size = tf.shape(src_emb)[0]
  per_sample_neg_num = tf.shape(neg_emb)[0] / batch_size
  pos_inner_product = sim_function(src_emb, pos_emb)

  src_emb_exp = tf.tile(tf.expand_dims(src_emb, axis=1),
                        [1, per_sample_neg_num, 1])
  src_emb_exp = tf.reshape(src_emb_exp, [-1, emb_dim])
  neg_inner_product = tf.reduce_sum(tf.multiply(src_emb_exp, neg_emb), axis=-1)

  logits = tf.concat([pos_inner_product, neg_inner_product], axis=0)
  labels = tf.concat([tf.ones_like(pos_inner_product),
                      -1 * tf.ones_like(neg_inner_product)], axis=0)

  loss = -tf.reduce_mean(tf.log_sigmoid(logits * labels))

  return [loss, logits, labels]


def triplet_loss(pos_src_emb, pos_edge_emb, neg_src_emb,
                 pos_dst_emb, neg_edge_emb, neg_dst_emb,
                 margin, neg_num, L=1):
  """triplet loss."""
  if L==2:
    pos_d = tf.reduce_sum(tf.square(pos_src_emb + pos_edge_emb - pos_dst_emb), axis=-1)
    neg_d = tf.reduce_sum(tf.square(neg_src_emb + neg_edge_emb - neg_dst_emb), axis=-1)
  else:
    pos_d = tf.reduce_sum(tf.abs(pos_src_emb + pos_edge_emb - pos_dst_emb), axis=-1)
    neg_d = tf.reduce_sum(tf.abs(neg_src_emb + neg_edge_emb - neg_dst_emb), axis=-1)
  if neg_num > 1:
    pos_d = tf.reshape(tf.tile(tf.expand_dims(pos_d, -1), [1, neg_num]), [-1])
  loss = tf.reduce_mean(tf.maximum(0.0, margin + pos_d - neg_d))
  return [loss, None, None]
