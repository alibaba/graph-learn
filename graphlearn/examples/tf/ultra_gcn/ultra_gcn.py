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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

import graphlearn as gl

EMB_PT_SIZE = 128 * 1024

class UltraGCN(object):
  def __init__(self, graph, batch_size, neg_num, neg_sampler,
      user_num, item_num, emb_dim, need_hash=False,  
      neg_weight=1.0, i2i_weight=1.0, l2_weight=0.0001, nbr_num=10, ps_num=0):
    self.graph = graph
    self.batch_size = batch_size
    self.emb_dim = emb_dim
    self.neg_num = neg_num
    self.nbr_num = nbr_num
    self.user_num = user_num
    self.item_num = item_num
    self.need_hash = need_hash
    self.i2i_weight = i2i_weight
    self.neg_weight = neg_weight
    self.l2_weight = l2_weight

    self.edge_sampler = graph.edge_sampler("u-i", batch_size, strategy="shuffle")
    self.neg_sampler = graph.negative_sampler("u-i", neg_num, neg_sampler)
    self.u_sampler = graph.node_sampler("u", batch_size, strategy="by_order")
    self.i_sampler = graph.node_sampler("i", batch_size, strategy="by_order")
    self.i2i_nbr_sampler = graph.neighbor_sampler("i-i", nbr_num, strategy="topk")

    self.train_iter = self.make_iterator(self.train_generator, 'train')
    self.u_iter = self.make_iterator(self.u_node_generator, 'save')
    self.i_iter = self.make_iterator(self.i_node_generator, 'save')

    # embedding
    partitioner = None
    if ps_num:
      partitioner = \
        tf.min_max_variable_partitioner(max_partitions=ps_num,
                                        min_slice_size=EMB_PT_SIZE)
    self.emb_table = {}
    with tf.variable_scope('ultra_gcn', reuse=tf.AUTO_REUSE):
      self.emb_table["user"] = tf.get_variable("user_emb",
                                               [user_num, emb_dim],
                                               trainable=True,
                                               partitioner=partitioner)
      self.emb_table["item"] = tf.get_variable("item_emb",
                                               [item_num, emb_dim],
                                               trainable=True,                                               
                                               partitioner=partitioner)

  def forward(self):
    values = self.train_iter.get_next()
    user_ids, user_degrees, item_ids, item_degrees,\
       nbr_ids, nbr_weights, neg_ids = values
    if self.need_hash:  # int->string->hash
      user_ids = tf.as_string(user_ids)
      user_ids = tf.strings.to_hash_bucket_fast(user_ids, self.user_num)
      item_ids = tf.as_string(item_ids)
      item_ids = tf.strings.to_hash_bucket_fast(item_ids, self.item_num)
      nbr_ids = tf.as_string(nbr_ids)
      nbr_ids = tf.strings.to_hash_bucket_fast(nbr_ids, self.item_num)
      neg_ids = tf.as_string(neg_ids)
      neg_ids = tf.strings.to_hash_bucket_fast(neg_ids, self.item_num)
    user_emb = tf.nn.embedding_lookup(self.emb_table["user"], user_ids)
    item_emb = tf.nn.embedding_lookup(self.emb_table["item"], item_ids)
    nbr_emb = tf.nn.embedding_lookup(self.emb_table["item"], nbr_ids)
    neg_emb = tf.nn.embedding_lookup(self.emb_table["item"], neg_ids)
    
    # UltraGCN base u2i
    pos_logit = tf.reduce_sum(user_emb * item_emb, axis=-1)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(pos_logit), logits=pos_logit)
    neg_logit = tf.reduce_sum(tf.expand_dims(user_emb, axis=1) * neg_emb, axis=-1)
    negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(neg_logit), logits=neg_logit)
    loss_u2i = tf.reduce_sum(true_xent * (1 + 1 / tf.sqrt(user_degrees * item_degrees))) \
      + self.neg_weight * tf.reduce_sum(tf.reduce_mean(negative_xent, axis=-1))
    # UltraGCN i2i
    nbr_logit = tf.reduce_sum(tf.expand_dims(user_emb, axis=1) * nbr_emb, axis=-1) # [batch_size, nbr_num]
    nbr_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(nbr_logit), logits=nbr_logit)  
    loss_i2i = tf.reduce_sum(nbr_xent * (1 + nbr_weights))
    # regularization
    loss_l2 = tf.nn.l2_loss(user_emb) + tf.nn.l2_loss(item_emb) +\
      tf.nn.l2_loss(nbr_emb) + tf.nn.l2_loss(neg_emb)
    return loss_u2i + self.i2i_weight * loss_i2i + self.l2_weight * loss_l2

  def user_emb(self):
    user_ids = self.u_iter.get_next()[0]
    ids = user_ids
    if self.need_hash:  # int->string->hash
      ids = tf.as_string(ids)
      ids = tf.strings.to_hash_bucket_fast(ids, self.user_num)
    user_emb = tf.nn.embedding_lookup(self.emb_table["user"], ids)
    return user_ids, user_emb

  def item_emb(self):
    item_ids = self.i_iter.get_next()[0]
    ids = item_ids
    if self.need_hash:  # int->string->hash
      ids = tf.as_string(ids)
      ids = tf.strings.to_hash_bucket_fast(ids, self.item_num)
    item_emb = tf.nn.embedding_lookup(self.emb_table["item"], ids)    
    return item_ids, item_emb

  def make_iterator(self, generator, mode='train'):
    if mode == 'train':
      output_types = [tf.int64, tf.float32, tf.int64, tf.float32,
        tf.int64, tf.float32, tf.int64]
      output_shapes = [tf.TensorShape([None]),
                       tf.TensorShape([None]),
                       tf.TensorShape([None]),
                       tf.TensorShape([None]),
                       tf.TensorShape([None, self.nbr_num]),
                       tf.TensorShape([None, self.nbr_num]),
                       tf.TensorShape([None, self.neg_num])]
    else:
      output_types = [tf.int64]
      output_shapes = [tf.TensorShape([None])]
    dataset = tf.data.Dataset.from_generator(generator,
                                             tuple(output_types),
                                             tuple(output_shapes)).prefetch(5)
    return dataset.make_initializable_iterator()

  def train_generator(self):
    while True:
      try:
        samples=[]
        edges = self.edge_sampler.get()
        neg_items = self.neg_sampler.get(edges.src_ids)
        nbr_items = self.i2i_nbr_sampler.get(edges.dst_ids)
        samples.append(edges.src_ids) # user ids
        samples.append(self.graph.out_degrees(edges.src_ids, 'u-i')) # user degrees
        samples.append(edges.dst_ids) # item ids
        samples.append(self.graph.out_degrees(edges.dst_ids, 'u-i_reverse')) # item degrees
        samples.append(nbr_items.layer_nodes(1).ids) # nbr item ids
        samples.append(nbr_items.layer_edges(1).weights) # nbr item weight.
        samples.append(neg_items.ids) # neg item ids
        yield(tuple(samples))
      except gl.OutOfRangeError:
        break
      
  def u_node_generator(self):
    while True:
      try:
        yield(tuple([self.u_sampler.get().ids]))
      except gl.OutOfRangeError:
        break

  def i_node_generator(self):
    while True:
      try:
        yield(tuple([self.i_sampler.get().ids]))
      except gl.OutOfRangeError:
        break