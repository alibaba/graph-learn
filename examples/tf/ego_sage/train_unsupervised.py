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
import graphlearn as gl
import tensorflow as tf
import graphlearn.python.nn.tf as tfg

from ego_sage import EgoGraphSAGE


def load_graph(config):
  data_dir = config['dataset_folder']
  g = gl.Graph() \
    .node(data_dir+'ogbl_collab_node', node_type='i',
          decoder=gl.Decoder(attr_types=['float'] * config['features_num'],
                             attr_dims=[0]*config['features_num'])) \
    .edge(data_dir+'ogbl_collab_train_edge', edge_type=('i', 'i', 'train'),
          decoder=gl.Decoder(weighted=True), directed=False)
  return g

def meta_path_sample(ego, ego_name, nbrs_num, sampler):
  """ creates the meta-math sampler of the input ego.
  config:
    ego: A query object, the input centric nodes/edges
    ego_name: A string, the name of `ego`.
    nbrs_num: A list, the number of neighbors for each hop.
    sampler: A string, the strategy of neighbor sampling.
  """
  alias_list = [ego_name + '_hop_' + str(i + 1) for i in range(len(nbrs_num))]
  for nbr_count, alias in zip(nbrs_num, alias_list):
    ego = ego.outV('train').sample(nbr_count).by(sampler).alias(alias)
  return ego

def query(graph, config):
  seed = graph.E('train').batch(config['batch_size']).shuffle(traverse=True)
  src = seed.outV().alias('src')
  dst = seed.inV().alias('dst')
  neg_dst = src.outNeg('train').sample(config['neg_num']).by(config['neg_sampler']).alias('neg_dst')
  src_ego = meta_path_sample(src, 'src', config['nbrs_num'], config['sampler'])
  dst_ego = meta_path_sample(dst, 'dst', config['nbrs_num'], config['sampler'])
  dst_neg_ego = meta_path_sample(neg_dst, 'neg_dst', config['nbrs_num'], config['sampler'])
  return seed.values()

def train(graph, model, config):
  tfg.conf.training = True
  query_train = query(graph, config)
  dataset = tfg.Dataset(query_train, window=5)
  src_ego = dataset.get_egograph('src')
  dst_ego = dataset.get_egograph('dst')
  neg_dst_ego = dataset.get_egograph('neg_dst')
  src_emb = model.forward(src_ego)
  dst_emb = model.forward(dst_ego)
  neg_dst_emb = model.forward(neg_dst_ego)
  # use sampled softmax loss with temperature.
  loss = tfg.unsupervised_softmax_cross_entropy_loss(src_emb, dst_emb, neg_dst_emb, 
    temperature=config['temperature'])
  return dataset.iterator, loss

def run(config):
  # graph input data
  g = load_graph(config=config)
  g.init()
  # Define Model
  dims = [config['features_num']] + [config['hidden_dim']] * (len(config['nbrs_num']) - 1) + [config['output_dim']]
  model = EgoGraphSAGE(dims,
                       agg_type=config['agg_type'],
                       dropout=config['drop_out'])
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
            'features_num': 128,
            'hidden_dim': 128,
            'output_dim': 128,
            'nbrs_num': [10, 5],
            'neg_num': 5,
            'learning_rate': 0.0001,
            'epoch': 1,
            'agg_type': 'mean',
            'drop_out': 0.0,
            'sampler': 'random',
            'neg_sampler': 'in_degree',
            'temperature': 0.07
            }
  run(config)