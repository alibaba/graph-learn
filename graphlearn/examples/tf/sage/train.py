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
from __future__ import print_function

import datetime

import numpy as np
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

import graphlearn as gl
import graphlearn.python.nn.tf as tfg

from edge_inducer import EdgeInducer

def load_graph(config):
  data_dir = config['dataset_folder']
  g = gl.Graph() \
    .node(data_dir+'ogbl_collab_node', node_type='i',
          decoder=gl.Decoder(attr_types=['float'] * config['features_num'],
                             attr_dims=[0]*config['features_num'])) \
    .edge(data_dir+'ogbl_collab_train_edge', edge_type=('i', 'i', 'train'),
          decoder=gl.Decoder(weighted=True), directed=False)
  return g


def train(g, model, config):
  seed = g.E('train').batch(config['batch_size']).shuffle(traverse=True)
  src = seed.outV().alias('pos_src')
  src.outV('train').sample(config['nbrs_num']).by('full').alias('src_hop1')
  dst = seed.inV().alias('pos_dst')
  dst.outV('train').sample(config['nbrs_num']).by('full').alias('dst_hop1')
  src.outNeg('train').sample(1).by('random').alias('neg_dst').\
    outV('train').sample(config['nbrs_num']).by('full').alias('neg_hop1')
  query = seed.values()
  dataset = tfg.Dataset(query, inducer=EdgeInducer(use_neg=True))
  pos_graph, neg_graph = dataset.get_batchgraph()
  pos_src, pos_dst = model.forward(batchgraph=pos_graph)
  neg_src, neg_dst = model.forward(batchgraph=neg_graph)
  pos_h = tf.reduce_sum(pos_src * pos_dst, axis=-1)
  neg_h = tf.reduce_sum(neg_src * neg_dst, axis=-1)
  loss = tfg.sigmoid_cross_entropy_loss(pos_h, neg_h)
  return dataset.iterator, loss

def run(config):
  # graph input data
  g = load_graph(config=config)
  g.init()
  # model
  model = tfg.GraphSAGE(config['batch_size'],
                        input_dim=config['features_num'],
                        hidden_dim=config['hidden_dim'],
                        output_dim=config['output_dim'],
                        depth=config['depth'],
                        drop_rate=config['drop_out'],
                        agg_type=config['agg_type'])
  # train
  iterator, loss = train(g, model, config)
  optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate'])
  train_op = optimizer.minimize(loss)
  train_ops = [loss, train_op]
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    step = 0
    print("Start Training...")
    for i in range(config['epoch']):
      try:
        while True:
          ret = sess.run(train_ops)
          print("Epoch {}, Iter {}, Loss {:.5f}".format(i, step, ret[0]))
          step += 1
      except tf.errors.OutOfRangeError:
        sess.run(iterator.initializer) # reinitialize dataset.
  g.close()

if __name__ == "__main__":
  config = {'dataset_folder': '../../data/ogbl_collab/',
            'batch_size': 128,
            'hidden_dim': 32,
            'output_dim': 32,
            'features_num': 128,
            'nbrs_num': 100,
            'depth': 3,
            'neg_num': 1,
            'learning_rate': 0.0001,
            'agg_type': 'mean',
            'drop_out': 0.0,
            'epoch': 1
           }
  run(config)