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
"""Evaluates Top k hitrate for transE on FB15k-237."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
import numpy as np
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf


def topk_hit_rate(head_embs,
                  relation_embs,
                  vocab_embs,
                  tail_ids,
                  k=10,
                  loss_type='l2',
                  dim=128):
  head_embs_ph = tf.placeholder(dtype=tf.float32,
                                shape=[len(head_embs), dim])
  relation_embs_ph = tf.placeholder(dtype=tf.float32,
                                    shape=[len(relation_embs), dim])
  vocab_embs_ph = tf.placeholder(dtype=tf.float32, shape=[len(vocab_embs), dim])

  if loss_type == 'l2':
    score = -tf.reduce_sum(
        tf.square(tf.tile(tf.expand_dims(head_embs_ph + relation_embs_ph, 1),
                          [1, len(vocab_embs), 1])
                  - tf.tile(tf.expand_dims(vocab_embs_ph, 0),
                            [len(head_embs), 1, 1])),
        -1)
  else:
    score = -tf.reduce_sum(
        tf.abs(tf.tile(tf.expand_dims(head_embs_ph + relation_embs_ph, 1),
                       [1, len(vocab_embs), 1])
               - tf.tile(tf.expand_dims(vocab_embs_ph, 0),
                         [len(head_embs), 1, 1])),
        -1)

  in_top_k = tf.nn.in_top_k(score, tf.constant(tail_ids, dtype=tf.int32), k=k)
  hitrate = tf.reduce_mean(tf.cast(in_top_k, dtype=tf.float32))
  with tf.Session() as sess:
    hr = sess.run([hitrate],
                  feed_dict={head_embs_ph: head_embs,
                             relation_embs_ph: relation_embs,
                             vocab_embs_ph: vocab_embs})
  return hr


def main():
  parser = ArgumentParser("Caculates top k hitrate on FB15k-237 data.")
  parser.add_argument("--dataset_dir", default='../../data/FB15k-237',
                      help="Path to directory containing the dataset.")
  parser.add_argument("--emb_dir", default='./',
                      help="Path of directory containing embeddings.")
  parser.add_argument("--k", default=10,
                      help="top num to evaluate hitrate.")
  parser.add_argument("--dim", default=128,
                      help="embedding dimension.")
  parser.add_argument("--loss_type", default='l2',
                      help="loss type, l1 or l2.")
  parser.add_argument("--batch_size", default=100,
                      help="batch size for evaluation.")

  args = parser.parse_args()
  dataset_dir = args.dataset_dir
  emb_dir = args.emb_dir

  # load data
  vocab_emb_path = emb_dir + '/id_entity.npy'
  gt_table_path = dataset_dir + '/test_tuple_table'
  relation_emb_path = emb_dir + '/id_relation.npy'

  vocab_emb = np.load(vocab_emb_path)
  id2idx = np.zeros(vocab_emb.shape[0], dtype=np.int32)
  cur = 0
  for i in vocab_emb[:, 0]:
    id2idx[int(i)] = cur
    cur += 1

  relation_emb = np.load(relation_emb_path)
  rid2idx = np.zeros(relation_emb.shape[0], dtype=np.int32)
  cur = 0
  for i in relation_emb[:, 0]:
    rid2idx[int(i)] = cur
    cur += 1

  with open(gt_table_path, 'r') as f:
    i = 0
    t_indices = []
    h_embs = []
    r_embs = []
    for line in f.readlines():
      if i == 0:
        i += 1
        continue  # skip head
      strs = line.strip().split('\t')
      h_embs.append(vocab_emb[id2idx[int(strs[0])]][1:])
      t_indices.append(id2idx[int(strs[1])])
      r_embs.append(relation_emb[rid2idx[int(strs[2])]][1:])

  # evaluation
  batch_size = args.batch_size
  epoch = len(t_indices) // batch_size
  total_hitrate = []
  for i in range(epoch):
    hitrate = topk_hit_rate(h_embs[i * batch_size: (i + 1) * batch_size],
                            r_embs[i * batch_size: (i + 1) * batch_size],
                            vocab_emb[:, 1:],
                            t_indices[i * batch_size: (i + 1) * batch_size],
                            k=args.k,
                            loss_type=args.loss_type,
                            dim=args.dim)
    total_hitrate.append(hitrate)
    print('hitrate', hitrate)

  hitrate = topk_hit_rate(h_embs[batch_size * epoch:],
                          r_embs[batch_size * epoch:],
                          vocab_emb[:, 1:],
                          t_indices[batch_size * epoch:],
                          k=args.k,
                          loss_type=args.loss_type,
                          dim=args.dim)
  total_hitrate.append(hitrate)
  print('hitrate', hitrate)
  print('avg hitrate', np.mean(total_hitrate))

if __name__ == "__main__":
  main()
