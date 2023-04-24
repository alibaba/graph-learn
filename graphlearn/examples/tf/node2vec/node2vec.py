# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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
import graphlearn.python.nn.tf as tfg

class Node2Vec(tfg.Module):
  def __init__(self,
               walk_len,
               l,
               r,
               bucket_size,
               need_hash,
               dim,
               **kwargs):
    """Node2Vec model.
    Args:
      walk_len: integer, random walk steps
      l: left window size of random walk pair generated.
      r: right window size of random walk pair generated.
      bucket_size: embedding bucket size
      dim: embedding dimension
    """
    self.walk_len = walk_len
    self.l = l
    self.r = r
    self.dim = dim

    self._target_encoder = tfg.EmbeddingColumn('src_emb', bucket_size, dim, need_hash)
    self._context_encoder = tfg.EmbeddingColumn('dst_emb', bucket_size, dim, need_hash)
    self._neg_num = kwargs.get('neg_num', 1)

  def _pair_indices(self):
    """ Generate indices of pairs from random walks, with left window
    size and right window size.
    """
    indices_src = []
    indices_dst = []
    for i in range(self.walk_len):
      for j in range(max(i - self.l, 0), i):
        indices_src.append(i)
        indices_dst.append(j)
      for j in range(i + 1, min(i + self.r + 1, self.walk_len)):
        indices_src.append(i)
        indices_dst.append(j)
    return indices_src, indices_dst

  def forward(self, x, neg=None):
    """ Encode tensor ids, if neg is None, encode x with target_encoder,
    else, encode x with both target_encoder and context_encoder, and encode
    neg with context_encoder, and return a tuple.
    """
    if neg is None:
      return self._target_encoder(tf.reshape(x, [-1]))
    else:
      return self._target_encoder(tf.reshape(x, [-1])), \
             self._context_encoder(tf.reshape(x, [-1])), \
             self._context_encoder(tf.reshape(neg, [-1]))

  def loss(self, walks_emb_src, walks_emb_dst, neg_emb):
    """ First gather embeddings from walks_emb_src and walks_emb_dst between
    window (left_window_size, right_window_size) as pairs, and then calculate
    pointwise sigmoid cross entropy loss.

    Args:
      walks_emb_src: embedding of walks encoded by target_encoder, shape=[-1, D]
      walks_emb_dst: embedding of walks encoded by context_encoder, shape=[-1, D]
      neg_emb: embedding of negative samples encoded by context_encoder, shape=[-1, D]
    """
    path_src = tf.reshape(walks_emb_src, shape=[-1, self.walk_len, self.dim])
    path_dst = tf.reshape(walks_emb_dst, shape=[-1, self.walk_len, self.dim])
    neg_dst = tf.reshape(neg_emb, shape=[-1, self._neg_num, self.dim])
    src_indices, dst_indices = self._pair_indices()

    src = tf.reshape(tf.gather(path_src, src_indices, axis=1),
                     shape=[-1, self.dim])
    dst = tf.reshape(tf.gather(path_dst, dst_indices, axis=1),
                     shape=[-1, self.dim])

    pos_logit = tf.reduce_sum(
      tf.multiply(src, dst, name='rank_dot_prod'), axis=-1)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(pos_logit), logits=pos_logit)
    neg_logit = tf.reduce_sum(
      tf.multiply(
        tf.tile(tf.expand_dims(walks_emb_src, axis=1), [1, self._neg_num, 1]),
        neg_dst, name='rank_dot_prod'),
      axis=-1)
    neg_xent = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.zeros_like(neg_logit), logits=neg_logit)
    lo = tf.reduce_mean(true_xent) + tf.reduce_mean(neg_xent)
    return lo

