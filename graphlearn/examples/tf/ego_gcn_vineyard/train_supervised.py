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

import json
import os

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
from graphlearn.examples.tf.ego_sage.ego_sage import EgoGraphSAGE
from graphlearn.examples.tf.ego_sage.ego_sage_data_loader import EgoSAGESupervisedDataLoader

flags = tf.app.flags
FLAGS = flags.FLAGS
# user-defined params
flags.DEFINE_integer('epochs', 2, 'training epochs')
flags.DEFINE_string('node_type', 'paper', 'node type')
flags.DEFINE_string('edge_type', 'cites', 'edge type')
flags.DEFINE_integer('class_num', 349, 'final output embedding dim')
flags.DEFINE_integer('features_num', 128, 'number of float attrs.')
flags.DEFINE_integer('hops_num', 2, 'number of float attrs.')
flags.DEFINE_string('nbrs_num', "[25, 10]", 'number of float attrs.')
flags.DEFINE_integer('hidden_dim', 128, 'hidden layer dim')
flags.DEFINE_float('in_drop_rate', 0.5, 'drop out rate')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')

flags.DEFINE_string('vineyard_socket', os.environ.get("VIHEYARD_IPC_SOCKET", "/tmp/vineyard.sock"), 'vineyard IPC socket location')
flags.DEFINE_integer('vineyard_fragment_id', -1, 'Object ID for vineyard fragment or vineyard fragment group')

nbrs_num = json.loads(FLAGS.nbrs_num)

def load_graph():
  import vineyard
  client = vineyard.connect(FLAGS.vineyard_socket)
  meta = client.get_meta(vineyard.ObjectID(FLAGS.vineyard_fragment_id))
  if meta.typename == 'vineyard::ArrowFragmentGroup':
     vineyard_fragment_id = int(meta['frag_object_id_0'].id)
  else:
     vineyard_fragment_id = int(meta.id)

  g = gl.Graph()
  g.vineyard(
     handle={
        'vineyard_id': vineyard_fragment_id,
        'vineyard_socket': FLAGS.vineyard_socket,
        'node_schema': ['paper:false:true:3:%d:0' % FLAGS.features_num],
        'edge_schema': ['paper:cites:paper:false:false:1:0:0'],
     },
     nodes=[FLAGS.node_type],
     edges=[[FLAGS.node_type, FLAGS.edge_type, FLAGS.node_type]],
  )

  features = ['feat_%d' % i for i in range(FLAGS.features_num)]
  g.node_attributes(FLAGS.node_type, features, 0, FLAGS.features_num, 0)
  g.edge_attributes(FLAGS.edge_type, [], 0, 0, 0)
  g.node_view(FLAGS.node_type, gl.Mask.TRAIN, 0, 100, (0, 75))
  g.node_view(FLAGS.node_type, gl.Mask.VAL, 0, 100, (75, 85))
  g.node_view(FLAGS.node_type, gl.Mask.TEST, 0, 100, (85, 100))
  return g

def main(unused_argv):
  g = load_graph()
  g.init()

  # Define Model
  dimensions = [FLAGS.features_num] + [FLAGS.hidden_dim] * (FLAGS.hops_num - 1) + [FLAGS.class_num]
  model = EgoGraphSAGE(dimensions, act_func=tf.nn.relu, dropout=FLAGS.in_drop_rate)

  # prepare train dataset
  train_data = EgoSAGESupervisedDataLoader(
    g, gl.Mask.TRAIN,
    node_type=FLAGS.node_type, edge_type=FLAGS.edge_type,
    nbrs_num=nbrs_num, hops_num=FLAGS.hops_num,
  )
  train_embedding = model.forward(train_data.src_ego)
  train_labels = train_data.src_ego.src.labels
  loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=train_labels, logits=train_embedding,
    )
  )
  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

  # prepare test dataset
  test_data = EgoSAGESupervisedDataLoader(
    g, gl.Mask.TEST,
    node_type=FLAGS.node_type, edge_type=FLAGS.edge_type,
    nbrs_num=nbrs_num, hops_num=FLAGS.hops_num,
  )
  test_embedding = model.forward(test_data.src_ego)
  test_labels = test_data.src_ego.src.labels
  test_indices = tf.math.argmax(test_embedding, 1, output_type=tf.int32)
  test_acc = tf.div(
    tf.reduce_sum(tf.cast(tf.math.equal(test_indices, test_labels), tf.float32)),
    tf.cast(tf.shape(test_labels)[0], tf.float32),
  )

  # train and test
  trainer = LocalTrainer()
  trainer.train(train_data.iterator, loss, optimizer, epochs=FLAGS.epochs)
  trainer.test(test_data.iterator, test_acc)

  # finish
  g.close()

if __name__ == "__main__":
  tf.app.run()
