# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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

import os
import sys

import numpy as np
import graphlearn as gl
import tensorflow as tf
import graphlearn.python.nn.tf as tfg

sys.path.append(os.path.dirname(sys.path[0]))
from node2vec import Node2Vec
from trainer import DistTrainer

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('task_index', None, 'Task index')
flags.DEFINE_integer('task_count', None, 'Task count')
flags.DEFINE_string('job_name', None, 'worker or ps')
flags.DEFINE_string('ps_hosts', '', 'ps hosts')
flags.DEFINE_string('worker_hosts', '', 'worker hosts')
flags.DEFINE_string('tables', '', 'input odps tables names of node and edge.')
flags.DEFINE_string('outputs', '', 'ouput odps tables name of node embeddings')
flags.DEFINE_string('buckets', '', 'oss buckets for ckpt')
flags.DEFINE_string('ckpt_dir', '', 'oss buckets for ckpt')
# user-defined params
flags.DEFINE_integer('epochs', 10, 'training epochs')
flags.DEFINE_integer('node_count', 10312, 'count of nodes')
flags.DEFINE_bool('need_hash', True, 'is node id continues or not,'
                  'if continues, need_hash is False and id should be in [0, node_count).')
flags.DEFINE_integer('batch_size', 128, 'minibatch size')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_integer('output_dim', 128, 'final output embedding dim')
flags.DEFINE_integer('neg_num', 10, 'negative sampling num')
flags.DEFINE_integer('walk_len', 10, 'random walk length')
flags.DEFINE_integer('window_size', 2, 'window size for pairing in random walks')
flags.DEFINE_bool('profiling', True, 'record timeline or not')
flags.DEFINE_float('param_return', 0.25, 'param return')
flags.DEFINE_float('param_inout', 0.25, 'param inout')
flags.DEFINE_integer('max_neighbor_count', 100000, 'max nerighbor count')

# This is a shared file system.
tracker = "/mnt/data/tracker"

# global settings.
gl.set_retry_times(30)
gl.set_timeout(300)
gl.set_tape_capacity(10)
gl.set_default_full_nbr_num(FLAGS.max_neighbor_count)
tfg.conf.emb_max_partitions = len(FLAGS.ps_hosts.split(',')) # embedding varible partition num.

def load_graph():
  node_table, edge_table = FLAGS.tables.split(',')[0:2]

  g = gl.Graph() \
    .node(node_table, node_type='i', decoder=gl.Decoder()) \
    .edge(edge_table, edge_type=('i', 'i', 'i-i'),
          decoder=gl.Decoder(weighted=True), directed=False)
  return g

def random_walk(graph):
  """ random walk along with i-i edges, and return walks and their negative neighbors
  as tensorflow tensors.
  Shape of returned walks is [B, W], and shape of their negative neighbors
  is [B, W, N]. While B=batch size, W=walk length, N=negtiave nums.
  """
  # B: batch_size, W: walk_len, N: neg_num
  p = FLAGS.param_return
  q = FLAGS.param_inout
  src = graph.V('i').batch(FLAGS.batch_size).shuffle(traverse=True).alias('src')
  walks = src.random_walk('i-i', FLAGS.walk_len - 1, p=p, q=q).alias('walks')

  neg = src.outNeg('i-i').sample(FLAGS.neg_num).by('random').alias('neg')
  walks_neg = walks.outNeg('i-i').sample(FLAGS.neg_num).by('random').alias('walks_neg')
  query = src.values()

  dataset = tfg.Dataset(query, 10)
  data_dict = dataset.get_data_dict()

  src = tf.expand_dims(data_dict['src'].ids, 1) # [B, 1]
  walks = tf.reshape(data_dict['walks'].ids, (-1, FLAGS.walk_len - 1)) # [B, W - 1]
  walks = tf.concat([src, walks], axis=1) # [B, W]

  neg = tf.expand_dims(tf.reshape(data_dict['neg'].ids,
                                  (-1, FLAGS.neg_num)), axis=1) # [B, 1, N]
  neg_dst_walks = tf.reshape(data_dict['walks_neg'].ids,
                             (-1, FLAGS.walk_len - 1, FLAGS.neg_num)) # # [B, W - 1, N]
  negs = tf.concat([neg, neg_dst_walks], axis=1) # [B, W, N]
  return dataset.iterator, walks, negs # [B, W], [B, W, N]

def train(graph, model):
  tfg.conf.training = True
  iterator, walks, negs = random_walk(graph)

  embs = model.forward(walks, negs)
  loss = model.loss(*embs)
  return iterator, loss

def save_node_embedding(graph, model):
  tfg.conf.training = False
  query_save = graph.V('i').batch(FLAGS.batch_size).alias('i').values()
  dataset = tfg.Dataset(query_save, 10)
  data_dict = dataset.get_data_dict()
  emb = model.forward(data_dict['i'].ids)
  return dataset.iterator, data_dict['i'].ids, emb

def main():
  worker_count = len(FLAGS.worker_hosts.split(','))
  ps_count = len(FLAGS.ps_hosts.split(','))
  graph_cluster = {'client_count': worker_count, 'server_count': ps_count, 'tracker': tracker}
  g = load_graph()
  if FLAGS.job_name == 'worker':
    g.init(cluster=graph_cluster, job_name='client', task_index=FLAGS.task_index)
  else:
    g.init(cluster=graph_cluster, job_name='server', task_index=FLAGS.task_index)

  ps_hosts = FLAGS.ps_hosts.split(',')
  worker_hosts = FLAGS.worker_hosts.split(',')
  tf_cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
  trainer = DistTrainer(tf_cluster,
                        FLAGS.job_name,
                        FLAGS.task_index,
                        worker_count,
                        ckpt_dir=FLAGS.buckets,
                        profiling=FLAGS.profiling)
  if FLAGS.job_name == 'worker':
    with trainer.context(): # model must under trainer.context.
      model = Node2Vec(FLAGS.walk_len,
                       FLAGS.window_size,
                       FLAGS.window_size,
                       FLAGS.node_count,
                       FLAGS.need_hash,
                       FLAGS.output_dim,
                       neg_num=FLAGS.neg_num)
      train_iterator, loss = train(g, model)
      save_iterator, ids, emb = save_node_embedding(g, model)
    # training
    trainer.train(train_iterator, loss, FLAGS.learning_rate, epochs=FLAGS.epochs)
    # saving node embedding.
    # save `ids`, `emb` in your own file system.
  else: 
    trainer.join()
  g.close()
  print('Finished!')

if __name__ == "__main__":
  main()
