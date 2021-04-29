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

import os, sys
import argparse
import datetime
import numpy as np
import tensorflow as tf

import graphlearn as gl
import graphlearn.python.nn.tf as tfg


def load_graph(args):
  dataset_folder = args.dataset_folder
  node_type = args.node_type
  edge_type = args.edge_type
  g = gl.Graph()                                                           \
        .node(dataset_folder + "node_table", node_type=node_type,
              decoder=gl.Decoder(labeled=True,
                                 attr_types=["float"] * args.features_num,
                                 attr_delimiter=":"))                      \
        .edge(dataset_folder + "edge_table",
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
  assert len(args.neighs_num) == args.hops_num
  prefix = ('train', 'test', 'val')[mask.value - 1]
  bs = getattr(args, prefix + '_batch_size')
  q = graph.V(args.node_type, mask=mask).batch(bs).alias(prefix)
  for idx, hop in enumerate(args.neighs_num):
    alias = prefix + '_hop' + str(idx)
    q = q.outV(args.edge_type).sample(hop).by('random').alias(alias)
  return q.values()

def supervised_loss(logits, labels):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  return tf.reduce_mean(loss)

def accuracy(logits, labels):
  indices = tf.math.argmax(logits, 1, output_type=tf.int32)
  correct = tf.reduce_sum(tf.cast(tf.math.equal(indices, labels), tf.float32))
  return correct / tf.cast(tf.shape(labels)[0], tf.float32)

def train(args, graph):
  # Train EgoGraph
  query_train = query(gl.Mask.TRAIN, args, graph)
  df_train = tfg.DataFlow(query_train, window=1)
  eg_train = df_train.get_ego_graph('train')

  # Test EgoGraph
  query_test = query(gl.Mask.TEST, args, graph)
  df_test = tfg.DataFlow(query_test, window=1)
  eg_test = df_test.get_ego_graph('test')

  # Define Model
  dims = [args.features_num] + [args.hidden_dim] * (args.hops_num - 1) \
        + [args.class_num]
  model = tfg.HomoEgoGraphSAGE(np.array(dims),
                               agg_type=args.agg_type,
                               com_type=args.com_type,
                               active_fn=tf.nn.relu,
                               dropout=args.in_drop_rate)

  train_embeddings = model.forward(eg_train)
  train_loss = supervised_loss(train_embeddings, eg_train.nodes.labels)

  trainer = tfg.Trainer(
      optimizer=tf.train.AdamOptimizer(learning_rate=args.learning_rate))
  trainer.minimize(train_loss)

  tfg.conf.training = False
  test_embeddings = model.forward(eg_test)
  test_acc = accuracy(test_embeddings, eg_test.nodes.labels)

  print("Start Trainging...")
  iter = [0]
  def print_loss(ret):
    if (iter[0] % 2 == 0):
      print("{} Iter {}, Loss: {:.5f}"
            .format(datetime.datetime.now(), iter[0], ret[0]))
    iter[0] += 1
  trainer.step_to_epochs(df_train, args.epoch, [train_loss], print_loss)

  print("Start Testing...")
  total_test_acc = []
  def print_acc(ret):
    total_test_acc.append(ret[0])
    print('Test Accuracy is: {:.4f}'.format(np.mean(total_test_acc)))
  trainer.run_one_epoch(df_test, [test_acc], print_acc)

  trainer.close()

def run(args):
  gl.set_tape_capacity(1)
  g = load_graph(args)
  g.init()
  train(args, g)
  g.close()


if __name__ == "__main__":
  """
  Data cora:
    Epochs=40, lr=0.05, Test accuracy=0.8140
  """
  cur_path = sys.path[0]
  argparser = argparse.ArgumentParser("Train EgoSAGE Supervised.")
  argparser.add_argument('--dataset_folder', type=str,
                         default=os.path.join(cur_path, '../../../data/cora/'),
                         help="Dataset Folder, list files are node_table, edge_table, "
                              "train_table, val_table and test_table")
  argparser.add_argument('--class_num', type=int, default=7)
  argparser.add_argument('--features_num', type=int, default=1433)
  argparser.add_argument('--train_batch_size', type=int, default=140)
  argparser.add_argument('--val_batch_size', type=int, default=300)
  argparser.add_argument('--test_batch_size', type=int, default=1000)
  argparser.add_argument('--hidden_dim', type=int, default=128)
  argparser.add_argument('--in_drop_rate', type=float, default=0.5)
  argparser.add_argument('--hops_num', type=int, default=2)
  argparser.add_argument('--neighs_num', type=list, default=[25, 10])
  argparser.add_argument('--agg_type', type=str, default="gcn")
  argparser.add_argument('--com_type', type=str, default="gcn")
  argparser.add_argument('--learning_algo', type=str, default="adam")
  argparser.add_argument('--learning_rate', type=float, default=0.05)
  argparser.add_argument('--weight_decay', type=float, default=0.0005)
  argparser.add_argument('--epoch', type=int, default=40)
  argparser.add_argument('--node_type', type=str, default='item')
  argparser.add_argument('--edge_type', type=str, default='relation')
  args = argparser.parse_args()

  run(args)
