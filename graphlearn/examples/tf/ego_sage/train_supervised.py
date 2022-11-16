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
from graphlearn.examples.tf.trainer import LocalTrainer

from ego_sage import EgoGraphSAGE
from ego_sage_data_loader import EgoSAGESupervisedDataLoader

def parse_args():
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
  argparser.add_argument('--hidden_dim', type=int, default=128)
  argparser.add_argument('--in_drop_rate', type=float, default=0.5)
  argparser.add_argument('--hops_num', type=int, default=2)
  argparser.add_argument('--nbrs_num', type=list, default=[25, 10])
  argparser.add_argument('--agg_type', type=str, default="gcn")
  argparser.add_argument('--learning_algo', type=str, default="adam")
  argparser.add_argument('--learning_rate', type=float, default=0.05)
  argparser.add_argument('--weight_decay', type=float, default=0.0005)
  argparser.add_argument('--epochs', type=int, default=40)
  argparser.add_argument('--node_type', type=str, default='item')
  argparser.add_argument('--edge_type', type=str, default='relation')
  return argparser.parse_args()

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

def supervised_loss(logits, labels):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  return tf.reduce_mean(loss)

def accuracy(logits, labels):
  indices = tf.math.argmax(logits, 1, output_type=tf.int32)
  correct = tf.reduce_sum(tf.cast(tf.math.equal(indices, labels), tf.float32))
  return correct / tf.cast(tf.shape(labels)[0], tf.float32)

def run(args):
  gl.set_tape_capacity(1)
  g = load_graph(args)
  g.init()
  # Define Model
  dims = [args.features_num] + [args.hidden_dim] * (args.hops_num - 1) \
        + [args.class_num]
  model = EgoGraphSAGE(dims,
                       agg_type=args.agg_type,
                       act_func=tf.nn.relu,
                       dropout=args.in_drop_rate)

  # prepare train dataset
  train_data = EgoSAGESupervisedDataLoader(g, gl.Mask.TRAIN, 'random', args.train_batch_size,
                                           node_type=args.node_type, edge_type=args.edge_type,
                                           nbrs_num=args.nbrs_num, hops_num=args.hops_num)
  train_embedding = model.forward(train_data.src_ego)
  loss = supervised_loss(train_embedding, train_data.src_ego.src.labels)
  optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

  # prepare test dataset
  test_data = EgoSAGESupervisedDataLoader(g, gl.Mask.TEST, 'random', args.test_batch_size,
                                          node_type=args.node_type, edge_type=args.edge_type,
                                          nbrs_num=args.nbrs_num, hops_num=args.hops_num)
  test_embedding = model.forward(test_data.src_ego)
  test_acc = accuracy(test_embedding, test_data.src_ego.src.labels)

  # train and test
  trainer = LocalTrainer()
  trainer.train(train_data.iterator, loss, optimizer, epochs=args.epochs)
  trainer.test(test_data.iterator, test_acc)

  # finish
  g.close()

if __name__ == "__main__":
  """
  Data cora:
    Epochs=40, lr=0.05, Test accuracy=0.8140
  """
  run(parse_args())
