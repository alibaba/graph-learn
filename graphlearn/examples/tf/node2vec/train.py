# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
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

import graphlearn as gl
import graphlearn.python.nn.tf as tfg
import numpy as np
import tensorflow as tf

from node2vec import Node2Vec
sys.path.append(os.path.dirname(sys.path[0]))

from tensorflow.python.client import timeline
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('tables', '../../data/blogcatelog/node_table,../../data/blogcatelog/edge_table',
                    'input odps tables names of node and edge.')
flags.DEFINE_string('ckpt_dir', './', 'dir for ckpt')
# user-defined params
flags.DEFINE_integer('epochs', 40, 'training epochs')
flags.DEFINE_integer('node_count', 10312, 'count of nodes')
flags.DEFINE_bool('need_hash', False, 'is node id continues or not,'
                  'if continues, need_hash is False and id should be in [0, node_count).')
flags.DEFINE_integer('batch_size', 256, 'minibatch size')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_integer('output_dim', 128, 'final output embedding dim')
flags.DEFINE_integer('neg_num', 10, 'negative sampling num')
flags.DEFINE_integer('walk_len', 20, 'random walk length')
flags.DEFINE_integer('window_size', 2, 'window size')
flags.DEFINE_float('param_return', 0.25, 'param return')
flags.DEFINE_float('param_inout', 0.25, 'param inout')

def load_graph():
  node_table, edge_table = FLAGS.tables.split(',')[0:2]

  g = gl.Graph() \
    .node(node_table, node_type='i',
          decoder=gl.Decoder(weighted=True)) \
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
  g = load_graph()
  g.init()

  model = Node2Vec(FLAGS.walk_len,
                   FLAGS.window_size,
                   FLAGS.window_size,
                   FLAGS.node_count,
                   FLAGS.need_hash,
                   FLAGS.output_dim,
                   neg_num=FLAGS.neg_num)
  train_iterator, loss = train(g, model)
  optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  train_op = optimizer.minimize(loss)
  train_ops = [loss, train_op]
  save_iterator, ids, emb = save_node_embedding(g, model)
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.initializer)
    step = 0
    print("Start Training...")
    for i in range(FLAGS.epochs):
      try:
        while True:
          if step < 20 and step > 10:
            ret = sess.run(train_ops,
                           options=run_options,
                           run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            content = tl.generate_chrome_trace_format()
            file_name = 'timeline_' + str(step) + '.json'
            save_path = os.path.join(FLAGS.ckpt_dir, file_name)
            writeGFile = tf.gfile.GFile(save_path, mode='w')
            writeGFile.write(content)
            writeGFile.flush()
            writeGFile.close()
          else:
            ret = sess.run(train_ops)
          print("Epoch {}, Iter {}, Loss {:.5f}".format(i, step, ret[0]))
          step += 1
      except tf.errors.OutOfRangeError:
        sess.run(train_iterator.initializer) # reinitialize dataset.

    sess.run(save_iterator.initializer)
    emb_set = []
    while True:
      try:
        rets = sess.run([ids, emb])
        emb_set.append(np.concatenate([np.reshape(rets[0], [-1, 1]), rets[1]],
                                      axis=1))
      except tf.errors.OutOfRangeError:
        print("Save node embedding finished.")
        break
    emb_set = np.concatenate(emb_set, axis=0)
    np.save('id_emb.npy', emb_set)
  g.close()

if __name__ == "__main__":
  main()
