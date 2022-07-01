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
import json
import sys
import time

import numpy as np
import graphlearn as gl
import tensorflow as tf
import graphlearn.python.nn.tf as tfg

sys.path.append("..")
from ego_sage import EgoGraphSAGE
from dist_trainer import DistTrainer


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('tables', '', 'input local files names of node and edge.')
flags.DEFINE_string('outputs', '', 'ouput local files name of node embeddings')
flags.DEFINE_string('ckpt_dir', None, 'checkpoint dir')
# user-defined params
flags.DEFINE_integer('epochs', 1, 'training epochs')
flags.DEFINE_integer('batch_size', 128, 'minibatch size')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_float('drop_out', 0.5, 'drop out rate')
flags.DEFINE_string('attr_types', '', 'attribute types, list')
flags.DEFINE_string('attr_dims', '', 'attribute dimension, list')
flags.DEFINE_integer('hidden_dim', 128, 'hidden layer dim')
flags.DEFINE_integer('output_dim', 128, 'final output embedding dim')
flags.DEFINE_integer('neg_num', 5, 'negative sampling num')
flags.DEFINE_string('nbrs_num', '[10,5]', 'string of list, neighbor num of each hop')
flags.DEFINE_string('agg_type', 'gcn', 'aggregation type for GraphSAGE, gcn, mean or sum')
flags.DEFINE_string('sampler', 'random', 'neighbor sampler strategy. random, in_degree, topk.')
flags.DEFINE_string('neg_sampler', 'random', 'negative sampler strategy. random, in_degree, node_weight.')
flags.DEFINE_float('temperature', 1.0, 'temperature of softmax loss.')

def load_graph(task_index):
  node_table, edge_table = FLAGS.tables.split(',')[0:2]
  attr_types = json.loads(FLAGS.attr_types)
  attr_dims = json.loads(FLAGS.attr_dims)
  g = gl.Graph() \
    .node(node_table + str(task_index), node_type='i',
          decoder=gl.Decoder(attr_types=attr_types,
                             attr_dims=attr_dims)) \
    .edge(edge_table + str(task_index), edge_type=('i', 'i', 'train'),
          decoder=gl.Decoder(weighted=True), directed=False)
  return g

def meta_path_sample(ego, ego_name, nbrs_num, sampler):
  """ creates the meta-math sampler of the input ego.
    ego: A query object, the input centric nodes/edges
    ego_name: A string, the name of `ego`.
    nbrs_num: A list, the number of neighbors for each hop.
    sampler: A string, the strategy of neighbor sampling.
  """
  alias_list = [ego_name + '_hop_' + str(i + 1) for i in range(len(nbrs_num))]
  for nbr_count, alias in zip(nbrs_num, alias_list):
    ego = ego.outV('train').sample(nbr_count).by(sampler).alias(alias)
  return ego

def query(graph):
  seed = graph.E('train').batch(FLAGS.batch_size).shuffle(traverse=True)
  src = seed.outV().alias('src')
  dst = seed.inV().alias('dst')
  neg_dst = src.outNeg('train').sample(FLAGS.neg_num).by(FLAGS.neg_sampler).alias('neg_dst')
  nbrs_num = json.loads(FLAGS.nbrs_num)
  src_ego = meta_path_sample(src, 'src', nbrs_num, FLAGS.sampler)
  dst_ego = meta_path_sample(dst, 'dst', nbrs_num, FLAGS.sampler)
  dst_neg_ego = meta_path_sample(neg_dst, 'neg_dst', nbrs_num, FLAGS.sampler)
  return seed.values()

def train(graph, model):
  tfg.conf.training = True
  query_train = query(graph)
  dataset = tfg.Dataset(query_train, window=1)
  src_ego = dataset.get_egograph('src')
  dst_ego = dataset.get_egograph('dst')
  neg_dst_ego = dataset.get_egograph('neg_dst')
  src_emb = model.forward(src_ego)
  dst_emb = model.forward(dst_ego)
  neg_dst_emb = model.forward(neg_dst_ego)
  # use sampled softmax loss with temperature.
  loss = tfg.unsupervised_softmax_cross_entropy_loss(src_emb, dst_emb, neg_dst_emb, 
    temperature=FLAGS.temperature)
  return dataset.iterator, loss

def save_node_embedding(graph, model):
  tfg.conf.training = False
  seed = graph.V('i').batch(FLAGS.batch_size).alias('i')
  nbrs_num = json.loads(FLAGS.nbrs_num)
  query_save = meta_path_sample(seed, 'i', nbrs_num, FLAGS.sampler).values()
  dataset = tfg.Dataset(query_save, window=1)
  ego_graph = dataset.get_egograph('i')
  emb = model.forward(ego_graph)
  return dataset.iterator, ego_graph.src.ids, emb

def main():
  gl.set_tracker_mode(0)
  gl_cluster, tf_cluster, job_name, task_index = gl.get_cluster()
  ps_hosts = tf_cluster.get("ps")
  gl_cluster["server"] = ",".join([host.split(":")[0] + ":8889" for host in ps_hosts])
  worker_count = len(tf_cluster["worker"])

  # global settings.
  tfg.conf.emb_max_partitions = len(ps_hosts) # embedding varible partition num.

  g = load_graph(task_index)

  tf_cluster = tf.train.ClusterSpec(tf_cluster)
  trainer = DistTrainer(tf_cluster, job_name, task_index, worker_count)

  if job_name == 'ps':
    g.init(cluster=gl_cluster, job_name='server', task_index=task_index)
    trainer.join()
    g.wait_for_close()

  else:
    g.init(cluster=gl_cluster, job_name='client', task_index=task_index)
    # training and save embedding.
    
    attr_dims = json.loads(FLAGS.attr_dims)
    input_dim = sum([1 if not i else i for i in attr_dims])
    depth = len(json.loads(FLAGS.nbrs_num))
    dims = [input_dim] + [FLAGS.hidden_dim] * (depth- 1) + [FLAGS.output_dim]
    with trainer.context(): # model must under trainer.context.
      model = EgoGraphSAGE(dims,
                          agg_type=FLAGS.agg_type,
                          dropout=FLAGS.drop_out)
      train_iter, loss = train(g, model)
      save_iter, ids, emb = save_node_embedding(g, model)
    # training
    trainer.train(train_iter, loss, FLAGS.learning_rate, epochs=FLAGS.epochs)
    # saving node embedding.
    print('Start saving node embedding...')
    trainer.save(FLAGS.outputs.split(',')[0] + str(task_index), save_iter, ids, emb, FLAGS.batch_size)
      
    g.close()
    print('Finished!')


if __name__ == "__main__":
  main()
