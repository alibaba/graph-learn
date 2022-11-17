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
import json
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

from ego_rgcn import EgoRGCN
from ego_rgcn_data_loader import EgoRGCNDataLoader

flags = tf.app.flags
FLAGS = flags.FLAGS
# user-defined params
flags.DEFINE_integer('epochs', 40, 'training epochs')
flags.DEFINE_integer('train_batch_size', 128, 'training minibatch size')
flags.DEFINE_integer('test_batch_size', 128, 'test minibatch size')
flags.DEFINE_float('learning_rate', 0.05, 'learning rate')
flags.DEFINE_float('drop_out', 0.5, 'drop out rate')
flags.DEFINE_integer('hidden_dim', 128, 'hidden layer dim')
flags.DEFINE_integer('class_num', 7, 'final output embedding dim')
flags.DEFINE_string('nbrs_num', '[20]', 'string of list, neighbor num of each hop')
flags.DEFINE_string('agg_type', 'mean', 'aggregation type, mean, max or sum')
flags.DEFINE_string('sampler', 'random', 'neighbor sampler strategy. random, in_degree, topk.')
flags.DEFINE_integer('num_relations', 2, 'number of relations')
flags.DEFINE_integer('num_bases', 1, 'number of bases to use for RGCNConv')
flags.DEFINE_integer('num_blocks', None, 'number of blocks to use for RGCNConv')
flags.DEFINE_string('attr_types', None, 'node attribute types')
flags.DEFINE_string('attr_dims', None, 'node attribute dimensions')
flags.DEFINE_integer('float_attr_num', 1433, 
  'number of float attrs. If there is only float attrs, we use this flag to instead of two above flags.')


if FLAGS.attr_types is not None and FLAGS.attr_dims is not None:
  attr_types = json.loads(FLAGS.attr_types)
  attr_dims = json.loads(FLAGS.attr_dims)
else:
  assert FLAGS.float_attr_num > 0
  attr_types = ['float'] * FLAGS.float_attr_num
  attr_dims = [0] * FLAGS.float_attr_num
nbrs_num = json.loads(FLAGS.nbrs_num)

def load_graph():
  """ Load node and edge data to build graph.
    Note that node_type must be "i", and edge_type must be "r_i", 
    the number of edge tables must be the same as FLAGS.num_relations.
  """
  cur_path = sys.path[0]
  dataset_folder = os.path.join(cur_path, '../../data/cora/')
  g = gl.Graph()\
        .node(dataset_folder + "node_table", node_type="i",
              decoder=gl.Decoder(labeled=True,
                                 attr_types=attr_types,
                                 attr_delimiter=":"))                      \
        .edge(dataset_folder + "edge_table",
              edge_type=("i", "i", "r_0"),
              decoder=gl.Decoder(weighted=True), directed=False)           \
        .edge(dataset_folder + "edge_table_with_self_loop",
              edge_type=("i", "i", "r_1"),
              decoder=gl.Decoder(weighted=True), directed=False)           \
        .node(dataset_folder + "train_table", node_type="i",
              decoder=gl.Decoder(weighted=True), mask=gl.Mask.TRAIN)       \
        .node(dataset_folder + "test_table", node_type="i",
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

def main(unused_argv):
  g = load_graph()
  g.init()
  # Define Model
  input_dim = sum([1 if not i else i for i in attr_dims])
  model = EgoRGCN(input_dim,
                  FLAGS.hidden_dim,
                  FLAGS.class_num,
                  len(nbrs_num),
                  FLAGS.num_relations,
                  FLAGS.num_bases,
                  FLAGS.num_blocks,
                  agg_type=FLAGS.agg_type,
                  dropout=FLAGS.drop_out)
  # prepare train dataset
  train_data = EgoRGCNDataLoader(g, gl.Mask.TRAIN, FLAGS.sampler, FLAGS.train_batch_size,
                                 node_type='i', nbrs_num=nbrs_num, num_relations=FLAGS.num_relations)
  train_embedding = model.forward(train_data.x_list(), nbrs_num)
  loss = supervised_loss(train_embedding, train_data.labels)
  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

  # prepare test dataset
  test_data = EgoRGCNDataLoader(g, gl.Mask.TEST, FLAGS.sampler, FLAGS.test_batch_size,
                                 node_type='i', nbrs_num=nbrs_num, num_relations=FLAGS.num_relations)
  test_embedding = model.forward(test_data.x_list(), nbrs_num)
  test_acc = accuracy(test_embedding, test_data.labels)

  # train and test
  trainer = LocalTrainer()
  trainer.train(train_data.iterator, loss, optimizer, epochs=FLAGS.epochs)
  trainer.test(test_data.iterator, test_acc)

  # finish
  g.close()


if __name__ == "__main__":
  tf.app.run()