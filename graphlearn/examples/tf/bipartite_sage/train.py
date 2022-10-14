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

from bipartite_sage import BipartiteGraphSAGE
from hetero_edge_inducer import HeteroEdgeInducer

# TODO(baole): use a benchmark u2i recommender dataset.
def load_graph(config):
  g = gl.Graph()\
        .node("../../data/u2i/u2i_node_attrs", node_type="i",
              decoder=gl.Decoder(attr_types=config['i_attr_types'], attr_dims=config['i_attr_dims']))\
        .node("../../data/u2i/u2i_node_attrs", node_type="u",
              decoder=gl.Decoder(attr_types=config['u_attr_types'], attr_dims=config['u_attr_dims']))\
        .edge("../../data/u2i/u2i_20200222_train", edge_type=("u", "i", "u-i"),
              decoder=gl.Decoder(weighted=True), directed=False)
  return g

def train(g, model, config):
  seed = g.E('u-i').batch(config['batch_size']).shuffle(traverse=True)
  src = seed.outV().alias('pos_src')
  # only support 1-hop now.
  src.outV('u-i').sample(config['nbrs_num']).by('full').alias('src_hop1')
  dst = seed.inV().alias('pos_dst')
  dst.inV('u-i').sample(config['nbrs_num']).by('full').alias('dst_hop1')
  src.outNeg('u-i').sample(1).by('random').alias('neg_dst').\
    outV('u-i').sample(config['nbrs_num']).by('full').alias('neg_hop1')
  query = seed.values()
  dataset = tfg.Dataset(query, inducer=HeteroEdgeInducer(use_neg=True, 
    edge_types=[('u', 'u-i', 'i'), ('i', 'u-i_reverse', 'u')]))
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
  u_input_dim = sum([1 if not i else i for i in config['u_attr_dims']])
  i_input_dim = sum([1 if not i else i for i in config['i_attr_dims']])
  model = BipartiteGraphSAGE(src_input_dim=u_input_dim,
                             dst_input_dim=i_input_dim,
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
  config = {'dataset_folder': '../../data/u2i/',
            'batch_size': 128,
            'hidden_dim': 32,
            'output_dim': 32,
            'u_attr_types': [("string", 10000)],
            'u_attr_dims': [128],
            'i_attr_types': [("string", 10000)],
            'i_attr_dims': [128],
            'nbrs_num': 100,
            'depth': 3,
            'neg_num': 1,
            'learning_rate': 0.001,
            'agg_type': 'mean',
            'drop_out': 0.0,
            'epoch': 1
           }
  run(config)