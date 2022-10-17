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

import argparse
import datetime
import os
import sys

import numpy as np
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

import graphlearn as gl
import graphlearn.python.nn.tf as tfg

from ego_gat import EgoGAT

def load_graph(args):
  dataset_folder = args.dataset_folder
  node_type = args.node_type
  edge_type = args.edge_type
  g = gl.Graph()                                                           \
        .node(dataset_folder + "node_table", node_type=node_type,
              decoder=gl.Decoder(labeled=True,
                                 attr_types=["float"] * args.features_num,
                                 attr_delimiter=":"))                      \
        .edge(dataset_folder + "edge_table_with_self_loop",
              edge_type=(node_type, node_type, edge_type),
              decoder=gl.Decoder(weighted=True), directed=False)           \
        .node(dataset_folder + "train_table", node_type=node_type,
              decoder=gl.Decoder(weighted=True), mask=gl.Mask.TRAIN)       \
        .node(dataset_folder + "val_table", node_type=node_type,
              decoder=gl.Decoder(weighted=True), mask=gl.Mask.VAL)         \
        .node(dataset_folder + "test_table", node_type=node_type,
              decoder=gl.Decoder(weighted=True), mask=gl.Mask.TEST)
  return g

def query(mask, args, graph):
  assert len(args.nbrs_num) == args.hops_num
  prefix = ('train', 'test', 'val')[mask.value - 1]
  bs = getattr(args, prefix + '_batch_size')
  q = graph.V(args.node_type, mask=mask).batch(bs).alias(prefix)
  for idx, hop in enumerate(args.nbrs_num):
    alias = prefix + '_hop' + str(idx)
    q = q.outV(args.edge_type).sample(hop).by(args.sample_strategy).alias(alias)
  return q.values()

def supervised_loss(logits, labels):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  return tf.reduce_mean(loss)

def accuracy(logits, labels):
  indices = tf.math.argmax(logits, 1, output_type=tf.int32)
  correct = tf.reduce_sum(tf.cast(tf.math.equal(indices, labels), tf.float32))
  return correct / tf.cast(tf.shape(labels)[0], tf.float32)

def train(graph, model, args):
  tfg.conf.training = True
  query_train = query(gl.Mask.TRAIN, args, graph)
  dataset = tfg.Dataset(query_train, window=10)
  eg_train = dataset.get_egograph('train')
  train_embeddings = model.forward(eg_train)
  loss = supervised_loss(train_embeddings, eg_train.src.labels)
  return dataset.iterator, loss

def test(graph, model, args):
  tfg.conf.training = False
  query_test = query(gl.Mask.TEST, args, graph)
  dataset = tfg.Dataset(query_test, window=10)
  eg_test = dataset.get_egograph('test')
  test_embeddings = model.forward(eg_test)
  test_acc = accuracy(test_embeddings, eg_test.src.labels)
  return dataset.iterator, test_acc

def run(args):
  gl.set_tape_capacity(1)
  g = load_graph(args)
  g.init()
  # Define Model
  dims = [args.features_num] + [args.hidden_dim] * (args.hops_num - 1) \
        + [args.class_num]
  model = EgoGAT(dims=dims, 
                 num_head=args.num_heads, 
                 act_func=tf.nn.relu, 
                 droput=args.in_drop_rate,
                 attn_dropout=args.attn_drop_rate)
  # train and test
  train_iterator, loss = train(g, model, args)
  optimizer=tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  train_op = optimizer.minimize(loss)
  train_ops = [loss, train_op]
  test_iterator, test_acc = test(g, model, args)
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.initializer)
    step = 0
    print("Start Training...")
    for i in range(args.epoch):
      try:
        while True:
          ret = sess.run(train_ops)
          print("Epoch {}, Iter {}, Loss {:.5f}".format(i, step, ret[0]))
          step += 1
      except tf.errors.OutOfRangeError:
        sess.run(train_iterator.initializer) # reinitialize dataset.
  
    print("Start Testing...")
    total_test_acc = []
    sess.run(test_iterator.initializer)
    try:
      while True:
        ret = sess.run(test_acc)
        total_test_acc.append(ret)
    except tf.errors.OutOfRangeError:
      print("Finished.")
    print('Test Accuracy is: {:.4f}'.format(np.mean(total_test_acc)))
  g.close()

if __name__ == "__main__":
  cur_path = sys.path[0]
  argparser = argparse.ArgumentParser("Train EgoSAGE Supervised.")
  argparser.add_argument('--dataset_folder', type=str,
                         default=os.path.join(cur_path, '../../data/cora/'),
                         help="Dataset Folder, list files are node_table, edge_table, "
                              "train_table, val_table and test_table")
  argparser.add_argument('--class_num', type=int, default=7)
  argparser.add_argument('--features_num', type=int, default=1433)
  argparser.add_argument('--train_batch_size', type=int, default=140)
  argparser.add_argument('--val_batch_size', type=int, default=300)
  argparser.add_argument('--test_batch_size', type=int, default=1000)
  argparser.add_argument('--hidden_dim', type=int, default=16)
  argparser.add_argument('--in_drop_rate', type=float, default=0.6)
  argparser.add_argument('--attn_drop_rate', type=float, default=0.6)
  argparser.add_argument('--num_heads', type=list, default=[8, 1])
  argparser.add_argument('--hops_num', type=int, default=2)
  argparser.add_argument('--sample_strategy', type=str, default="random")
  argparser.add_argument('--nbrs_num', type=list, default=[5, 2])
  argparser.add_argument('--learning_algo', type=str, default="adamW")
  argparser.add_argument('--learning_rate', type=float, default=0.01)
  argparser.add_argument('--weight_decay', type=float, default=0.0005)
  argparser.add_argument('--epoch', type=int, default=200)
  argparser.add_argument('--node_type', type=str, default='item')
  argparser.add_argument('--edge_type', type=str, default='relation')
  args = argparser.parse_args()

  run(args)
