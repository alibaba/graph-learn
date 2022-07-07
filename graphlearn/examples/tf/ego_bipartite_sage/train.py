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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import tensorflow as tf
import graphlearn as gl
import graphlearn.python.nn.tf as tfg

from ego_bipartite_sage import EgoBipartiteGraphSAGE

# TODO(baole): use a benchmark book_data recommender dataset.
def load_graph(config):
  g = gl.Graph()\
        .node("../../data/books_data/gl_item.txt", node_type="i",
              decoder=gl.Decoder(attr_types=config['i_attr_types'], 
                                 attr_dims=config['i_attr_dims']))\
        .node("../../data/books_data/gl_user.txt", node_type="u",
              decoder=gl.Decoder(attr_types=config['u_attr_types'], 
                                 attr_dims=config['u_attr_dims']))\
        .edge("../../data/books_data/gl_train.txt", edge_type=("u", "i", "u-i"),
              decoder=gl.Decoder(weighted=False), directed=False)
  return g

def meta_path_sample(ego, ego_type, ego_name, nbrs_num, sampler):
  """ creates the meta-math sampler of the input ego.
    ego: A query object, the input centric nodes/edges
    ego_type: A string, the type of `ego`, 'u' or 'i'.
    ego_name: A string, the name of `ego`.
    nbrs_num: A list, the number of neighbors for each hop.
    sampler: A string, the strategy of neighbor sampling.
  """
  choice = int(ego_type == 'i')
  # for u is u-i-u-i-u..., for i is i-u-i-u-i....
  meta_path = [('outV', 'inV')[(i + choice) % 2] for i in range(len(nbrs_num))]
  alias_list = [ego_name + '_hop_' + str(i + 1) for i in range(len(nbrs_num))]
  for path, nbr_count, alias in zip(meta_path, nbrs_num, alias_list):
    ego = getattr(ego, path)('u-i').sample(nbr_count).by(sampler).alias(alias)
  return ego

def query(graph, config):
  # traverse graph to get positive and negative (u,i) samples.
  edge = graph.E('u-i').batch(config['batch_size']).alias('seed')
  src = edge.outV().alias('src')
  dst = edge.inV().alias('dst')
  neg_dst = src.outNeg('u-i').sample(config['neg_num']).by(config['neg_sampler']).alias('neg_dst')
  # meta-path sampling.
  src_ego = meta_path_sample(src, 'u', 'src', config['u_nbrs_num'], config['sampler'])
  dst_ego = meta_path_sample(dst, 'i', 'dst', config['i_nbrs_num'], config['sampler'])
  dst_neg_ego = meta_path_sample(neg_dst, 'i', 'neg_dst', config['u_nbrs_num'], config['sampler'])
  return edge.values()

def train(graph, src_model, dst_model, config):
  tfg.conf.training = True
  query_train = query(graph, config)
  dataset = tfg.Dataset(query_train, window=10)
  src_ego = dataset.get_egograph('src')
  dst_ego = dataset.get_egograph('dst')
  neg_dst_ego = dataset.get_egograph('neg_dst')
  src_emb = src_model.forward(src_ego)
  dst_emb = dst_model.forward(dst_ego)
  neg_dst_emb = dst_model.forward(neg_dst_ego)
  # use sampled softmax loss with temperature.
  loss = tfg.unsupervised_softmax_cross_entropy_loss(src_emb, dst_emb, neg_dst_emb, 
    temperature=config['temperature'])
  return dataset.iterator, loss

def node_embedding(graph, model, node_type, config):
  """ save node embedding.
  Args:
    node_type: 'u' or 'i'.
  Return:
    iterator, ids, embedding.
  """
  tfg.conf.training = False
  ego_name = 'save_node_'+node_type
  seed = graph.V(node_type).batch(config['batch_size']).alias(ego_name)
  nbrs_num = config['u_nbrs_num'] if node_type == 'u' else config['i_nbrs_num']
  query_save = meta_path_sample(seed, node_type, ego_name, nbrs_num, config['sampler']).values()
  dataset = tfg.Dataset(query_save, window=10)
  ego_graph = dataset.get_egograph(ego_name)
  emb = model.forward(ego_graph)
  return dataset.iterator, ego_graph.src.ids, emb

def dump_embedding(sess, iter, ids, emb, emb_writer):
  sess.run(iter.initializer)
  while True:
    try:
      outs = sess.run([ids, emb])
      # [B,], [B,dim]
      feat = [','.join(str(x) for x in arr) for arr in outs[1]]
      for id, feat in zip(outs[0], feat):
        emb_writer.write('%d\t%s\n' % (id, feat))
    except tf.errors.OutOfRangeError:
      print('Save node embeddings done.')
      break

def run(config):
  g = load_graph(config)
  g.init()
  # Define Model
  u_input_dim = sum([1 if not i else i for i in config['u_attr_dims']])
  u_hidden_dims = [config['hidden_dim']] * (len(config['u_nbrs_num']) - 1) + [config['output_dim']]
  i_input_dim = sum([1 if not i else i for i in config['i_attr_dims']])
  i_hidden_dims = [config['hidden_dim']] * (len(config['i_nbrs_num']) - 1) + [config['output_dim']]
  # two tower model for u and i.
  u_model = EgoBipartiteGraphSAGE(u_input_dim,
                                  i_input_dim,
                                  u_hidden_dims,
                                  agg_type=config['agg_type'],
                                  dropout=config['drop_out'])
  i_model = EgoBipartiteGraphSAGE(i_input_dim,
                                  u_input_dim,
                                  i_hidden_dims,
                                  agg_type=config['agg_type'],
                                  dropout=config['drop_out'])
  # train and save node embeddings.
  train_iterator, loss = train(g, u_model, i_model, config)
  u_save_iter, u_ids, u_emb = node_embedding(g, u_model, 'u', config)
  i_save_iter, i_ids, i_emb = node_embedding(g, i_model, 'i', config)
  optimizer=tf.train.AdamOptimizer(learning_rate=config['learning_rate'])
  train_op = optimizer.minimize(loss)
  train_ops = [loss, train_op]
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.initializer)
    step = 0
    print("Start Training...")
    for i in range(config['epoch']):
      try:
        while True:
          ret = sess.run(train_ops)
          if step % 100 == 0:
            print("Epoch {}, Iter {}, Loss {:.5f}".format(i, step, ret[0]))
          step += 1
      except tf.errors.OutOfRangeError:
        sess.run(train_iterator.initializer) # reinitialize dataset.
    print("Start saving embeddings...")
    u_emb_writer = open('u_emb.txt', 'w')
    i_emb_writer = open('i_emb.txt', 'w')
    u_emb_writer.write('id:int64\temb:string\n')
    i_emb_writer.write('id:int64\temb:string\n')
    dump_embedding(sess, u_save_iter, u_ids, u_emb, u_emb_writer)
    dump_embedding(sess, i_save_iter, i_ids, i_emb, i_emb_writer)
  g.close()


if __name__ == "__main__":
  config = {'batch_size': 512,
            'u_attr_types': [("int", 53000)],
            'u_attr_dims': [128],
            'i_attr_types': [("int", 92000)],
            'i_attr_dims': [128],
            'hidden_dim': 128,
            'output_dim': 128,
            'u_nbrs_num': [10, 5],
            'i_nbrs_num': [10, 5],
            'neg_num': 5,
            'learning_rate': 0.0001,
            'epoch': 10,
            'agg_type': 'mean',
            'drop_out': 0.0,
            'sampler': 'random',
            'neg_sampler': 'random',
            'temperature': 0.07
            }
  run(config)