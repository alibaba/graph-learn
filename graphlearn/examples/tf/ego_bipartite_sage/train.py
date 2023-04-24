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

import argparse
import datetime
import numpy as np
import os, sys

try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

import graphlearn as gl
import graphlearn.python.nn.tf as tfg

from ego_bipartite_sage import EgoBipartiteGraphSAGE

def load_graph(args):
  g = gl.Graph()\
        .node(args.item_path, node_type="i",
              decoder=gl.Decoder(attr_types=args.i_attr_types,
                                 attr_dims=args.i_attr_dims))\
        .node(args.user_path, node_type="u",
              decoder=gl.Decoder(attr_types=args.u_attr_types,
                                 attr_dims=args.u_attr_dims))\
        .edge(args.u2i_path, edge_type=("u", "i", "u-i"),
              decoder=gl.Decoder(weighted=args.edge_weighted), directed=False)
  if args.i2i_path != "":
    g.edge(args.i2i_path, edge_type=("i", "i", "i-i"),
              decoder=gl.Decoder(weighted=args.edge_weighted), directed=False)
  return g

def meta_path_sample(ego, ego_type, ego_name, nbrs_num, sampler, i2i):
  """ creates the meta-math sampler of the input ego.
    ego: A query object, the input centric nodes/edges
    ego_type: A string, the type of `ego`, 'u' or 'i'.
    ego_name: A string, the name of `ego`.
    nbrs_num: A list, the number of neighbors for each hop.
    sampler: A string, the strategy of neighbor sampling.
    i2i: Boolean, is i2i egde exist or not.
  """
  choice = int(ego_type == 'i')
  meta_path = []
  hops = range(len(nbrs_num))
  if i2i:
    # for u is u-i-i-i..., for i is i-i-i-i...
    meta_path = ['outV' for i in hops]
  else:
    # for u is u-i-u-i-u..., for i is i-u-i-u-i....
    meta_path = [('outV', 'inV')[(i + choice) % 2] for i in hops]
  alias_list = [ego_name + '_hop_' + str(i + 1) for i in hops]
  idx = 0
  mata_path_string = ""
  for path, nbr_count, alias in zip(meta_path, nbrs_num, alias_list):
    etype = ('u-i', 'i-i')[(int(i2i) and choice ) or (int(i2i) and not choice and idx > 0)]
    idx += 1
    mata_path_string += path + '(' + etype + ').'
    ego = getattr(ego, path)(etype).sample(nbr_count).by(sampler).alias(alias)
  print("Sampling meta path for {} is {}.".format(ego_type, mata_path_string))
  return ego

def query(graph, args):
  # traverse graph to get positive and negative (u,i) samples.
  edge = graph.E('u-i').batch(args.batch_size).alias('seed')
  src = edge.outV().alias('src')
  dst = edge.inV().alias('dst')
  neg_dst = src.outNeg('u-i').sample(args.neg_num).by(args.neg_sampler).alias('neg_dst')
  # meta-path sampling.
  src_ego = meta_path_sample(src, 'u', 'src', args.u_nbrs_num, args.sampler, args.i2i_path != "")
  dst_ego = meta_path_sample(dst, 'i', 'dst', args.i_nbrs_num, args.sampler, args.i2i_path != "")
  dst_neg_ego = meta_path_sample(neg_dst, 'i', 'neg_dst', args.u_nbrs_num, args.sampler, args.i2i_path != "")
  return edge.values()

def train(graph, src_model, dst_model, args):
  tfg.conf.training = True
  query_train = query(graph, args)
  dataset = tfg.Dataset(query_train, window=10)
  src_ego = dataset.get_egograph('src')
  dst_ego = dataset.get_egograph('dst')
  neg_dst_ego = dataset.get_egograph('neg_dst')
  src_emb = src_model.forward(src_ego)
  output_embeddings = tf.identity(src_emb, name="output_embeddings")
  dst_emb = dst_model.forward(dst_ego)
  neg_dst_emb = dst_model.forward(neg_dst_ego)
  # use sampled softmax loss with temperature.
  loss = tfg.unsupervised_softmax_cross_entropy_loss(output_embeddings, dst_emb, neg_dst_emb,
    temperature=args.temperature)
  return dataset.iterator, loss

def node_embedding(graph, model, node_type, args):
  """ save node embedding.
  Args:
    node_type: 'u' or 'i'.
  Return:
    iterator, ids, embedding.
  """
  tfg.conf.training = False
  ego_name = 'save_node_'+node_type
  seed = graph.V(node_type).batch(args.batch_size).alias(ego_name)
  nbrs_num = args.u_nbrs_num if node_type == 'u' else args.i_nbrs_num
  query_save = meta_path_sample(seed, node_type, ego_name, nbrs_num, args.sampler, args.i2i_path != "").values()
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

def run(args):
  g = load_graph(args)
  g.init()
  # Define Model
  u_input_dim = sum([1 if not i else i for i in args.u_attr_dims])
  u_hidden_dims = [args.hidden_dim] * (len(args.u_nbrs_num) - 1) + [args.output_dim]
  i_input_dim = sum([1 if not i else i for i in args.i_attr_dims])
  i_hidden_dims = [args.hidden_dim] * (len(args.i_nbrs_num) - 1) + [args.output_dim]
  # two tower model for u and i.
  u_model = EgoBipartiteGraphSAGE('src',
                                  u_input_dim,
                                  i_input_dim,
                                  u_hidden_dims,
                                  agg_type=args.agg_type,
                                  dropout=args.drop_out,
                                  i2i=args.i2i_path != "")
  dst_input_dim = i_input_dim if args.i2i_path != "" else u_input_dim
  i_model = EgoBipartiteGraphSAGE('dst',
                                  i_input_dim,
                                  dst_input_dim,
                                  i_hidden_dims,
                                  agg_type=args.agg_type,
                                  dropout=args.drop_out,
                                  i2i=args.i2i_path != "")
  # train and save node embeddings.
  train_iterator, loss = train(g, u_model, i_model, args)
  u_save_iter, u_ids, u_emb = node_embedding(g, u_model, 'u', args)
  i_save_iter, i_ids, i_emb = node_embedding(g, i_model, 'i', args)
  optimizer=tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  train_op = optimizer.minimize(loss)
  train_ops = [loss, train_op]
  saver = tf.train.Saver()
  with tf.Session() as sess:
    writer=tf.summary.FileWriter('./tensorboard', sess.graph)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.initializer)
    step = 0
    print("Start Training...")
    for i in range(args.epoch):
      try:
        while True:
          ret = sess.run(train_ops)
          if step % 100 == 0:
            print("Epoch {}, Iter {}, Loss {:.5f}".format(i, step, ret[0]))
          step += 1
      except tf.errors.OutOfRangeError:
        sess.run(train_iterator.initializer) # reinitialize dataset.

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt/model")
    print("Start saving checkpoint in {} ...".format(model_path))
    saver.save(sess, model_path)
    writer.close()

    print("Start saving embeddings...")
    u_emb_writer = open('u_emb.txt', 'w')
    i_emb_writer = open('i_emb.txt', 'w')
    u_emb_writer.write('id:int64\temb:string\n')
    i_emb_writer.write('id:int64\temb:string\n')
    dump_embedding(sess, u_save_iter, u_ids, u_emb, u_emb_writer)
    dump_embedding(sess, i_save_iter, i_ids, i_emb, i_emb_writer)
  g.close()


if __name__ == "__main__":
  """
  Default is an exmaple with only u2i edges.
  """
  cur_path = sys.path[0]
  argparser = argparse.ArgumentParser("Train EgoBipartiteSAGE.")
  import json

  argparser.add_argument('--user_path', type=str,
                         default=os.path.join(cur_path, '../../data/books_data/gl_user.txt'),
                         help="User attributes table path")
  argparser.add_argument('--item_path', type=str,
                         default=os.path.join(cur_path, '../../data/books_data/gl_item.txt'),
                         help="Item attributes table path")
  argparser.add_argument('--u2i_path', type=str,
                         default=os.path.join(cur_path, '../../data/books_data/gl_train.txt'),
                         help="U2i edge table path")
  argparser.add_argument('--i2i_path', type=str,
                         default="", help="I2i edge table path, this is optional")
  argparser.add_argument('--batch_size', type=int, default=512)
  argparser.add_argument('--u_attr_types',  type=json.loads, default='["int,53000"]')
  argparser.add_argument('--u_attr_dims', type=json.loads, default='[128]')
  argparser.add_argument('--i_attr_types', type=json.loads, default='["int,92000"]')
  argparser.add_argument('--i_attr_dims', type=json.loads, default='[128]')
  argparser.add_argument('--hidden_dim', type=int, default=128)
  argparser.add_argument('--output_dim', type=int, default=128)
  argparser.add_argument('--u_nbrs_num', nargs='+', type=int, default=[10, 5])
  argparser.add_argument('--i_nbrs_num', nargs='+', type=int, default=[10, 5])
  argparser.add_argument('--neg_num', type=int, default=5)
  argparser.add_argument('--learning_rate', type=float, default=0.0001)
  argparser.add_argument('--epoch', type=int, default=10)
  argparser.add_argument('--agg_type', type=str, default="mean")
  argparser.add_argument('--drop_out', type=float, default=0.0)
  argparser.add_argument('--sampler', type=str, default="random")
  argparser.add_argument('--neg_sampler', type=str, default="random")
  argparser.add_argument('--temperature', type=float, default=0.07)
  argparser.add_argument('--edge_weighted', action="store_true")
  args = argparser.parse_args()

  def parse_attr_type(value):
    v = value.split(',')
    if len(v) == 2:
      return (v[0], int(v[1]))
    else:
      assert len(v) == 1
      return v[0]

  args.u_attr_types = [parse_attr_type(v) for v in args.u_attr_types]
  args.i_attr_types = [parse_attr_type(v) for v in args.i_attr_types]

  print(args.u_attr_types, args.u_attr_dims, args.i_attr_types, args.i_attr_dims)
  run(args)