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

import argparse
import os, sys
import numpy as np
import graphlearn as gl
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf
import graphlearn.python.nn.tf as tfg

from ego_tgat import EgoTGAT, LinkScorePredict


def load_graph(args):
  features_num = args.features_num
  data_folder = args.data_folder
  srcf, dstf, edgef, trainf, valf, testf = [
    os.path.join(data_folder, f) for f in args.files.split(',')]

  src_decoder = gl.Decoder(
    attr_types=[('int', args.src_num)], attr_dims=[features_num])
  dst_decoder = gl.Decoder(
    attr_types=[('int', args.dst_num)], attr_dims=[features_num])
  edge_decoder = gl.Decoder(
    attr_types=["float"] * features_num, timestamped=True)

  g = gl.Graph()
  g = g.node(srcf, 'src', decoder=src_decoder) \
       .node(dstf, 'dst', decoder=dst_decoder) \
       .edge(edgef, edge_type=("src", "dst", "interaction"),
             decoder=edge_decoder, directed=False) \
       .edge(trainf, edge_type=("src", "dst", "train"),
             decoder=edge_decoder, directed=False) \
       .edge(valf, edge_type=("src", "dst", "val"),
             decoder=edge_decoder, directed=False) \
       .edge(testf, edge_type=("src", "dst", "test"),
             decoder=edge_decoder, directed=False)
  return g

def query(graph, args, etype='train'):
  """ Traverse events with asscending timestamps, and sample topk-timestamp
  neighbors for source, positive destination and negtaive destination of the
  events respectively.
  """
  events = graph.E(etype).batch(args.batch_size).alias("event")

  def sample_pos_neg_nbrs(events):
    srcV = events.outV().alias('src')
    dstV = events.inV().alias('dst')
    negV = srcV.outNeg("interaction").sample(1).by("random").alias("neg")
    nbrs = args.nbr_size
    def hops(src, name):
      for i, nbr in enumerate(nbrs):
        src = src.outE("interaction").sample(nbr).by("topk").alias(name + "_nbr_" + str(i + 1))\
                 .inV().alias(name + "_nbr_node_" + str(i + 1))
    hops(srcV, 'src')
    hops(dstV, 'dst')
    hops(negV, 'neg')
  sample_pos_neg_nbrs(events)
  return events.values()

def train(graph, model, score_pred, args):
  tfg.conf.training = True
  query_train = query(graph, args)
  dataset = tfg.Dataset(query_train)
  src_temporal = dataset.get_temporalgraph(
    'src', ['src_nbr_1', 'src_nbr_2'], ['src_nbr_node_1', 'src_nbr_node_2'], args.time_dim)
  dst_temporal = dataset.get_temporalgraph(
    'dst', ['dst_nbr_1', 'dst_nbr_2'], ['dst_nbr_node_1', 'dst_nbr_node_2'], args.time_dim)
  neg_dst_temporal = dataset.get_temporalgraph(
    'neg', ['neg_nbr_1', 'neg_nbr_2'], ['neg_nbr_node_1', 'neg_nbr_node_2'], args.time_dim)
  src_emb = model.forward(src_temporal)
  dst_emb = model.forward(dst_temporal)
  neg_dst_emb = model.forward(neg_dst_temporal)

  pos_score, neg_score = score_pred(src_emb, dst_emb, neg_dst_emb)
  loss = tfg.sigmoid_cross_entropy_loss(pos_score, neg_score)
  return dataset.iterator, loss

def test(graph, model, score_pred, args):
  tfg.conf.training = False
  query_test = query(graph, args, etype='test')
  dataset = tfg.Dataset(query_test)
  src_temporal = dataset.get_temporalgraph(
    'src', ['src_nbr_1', 'src_nbr_2'], ['src_nbr_node_1', 'src_nbr_node_2'], args.time_dim)
  dst_temporal = dataset.get_temporalgraph(
    'dst', ['dst_nbr_1', 'dst_nbr_2'], ['dst_nbr_node_1', 'dst_nbr_node_2'], args.time_dim)
  neg_dst_temporal = dataset.get_temporalgraph(
    'neg', ['neg_nbr_1', 'neg_nbr_2'], ['neg_nbr_node_1', 'neg_nbr_node_2'], args.time_dim)
  src_emb = model.forward(src_temporal)
  dst_emb = model.forward(dst_temporal)
  neg_dst_emb = model.forward(neg_dst_temporal)
  pos_score, neg_score = score_pred(src_emb, dst_emb, neg_dst_emb)

  pos_score = tf.nn.sigmoid(pos_score)
  neg_score = tf.nn.sigmoid(neg_score)
  correct = tf.concat(
    [tf.math.greater(pos_score, 0.5), tf.math.less(neg_score, 0.5)], axis=-1)
  correct = tf.cast(correct, tf.float32)
  correct = tf.reduce_sum(correct)
  test_acc = correct / tf.cast(tf.shape(dst_emb)[0], tf.float32) / 2
  return dataset.iterator, test_acc

def run(args):
  # graph input data
  g = load_graph(args)
  g.init()
  # Define Model

  nbr_dim = args.features_num * 2 + args.time_dim
  # Nbr nodes features, nbr edges features and timesatmps

  src_dim = args.features_num + args.time_dim
  # Src nodes features and timesatmps

  dims = [[[src_dim, nbr_dim], args.hidden_dim], [args.hidden_dim + args.time_dim, args.features_num]]

  model = EgoTGAT(dims,
                  num_head=args.num_heads,
                  act_func=tf.nn.relu)
  score_pred = LinkScorePredict(args.features_num)
  # train
  iterator, loss = train(g, model, score_pred, args)
  optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
  train_op = optimizer.minimize(loss)
  train_ops = [loss, train_op]

  test_iterator, test_acc = test(g, model, score_pred, args)
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    step = 0
    print("Start Training...")
    total_loss = []
    for i in range(args.epoch):
      try:
        while True:
          ret = sess.run(train_ops)
          total_loss.append(ret[0])
          if step % 100 == 0:
            print("Epoch {}, Iter {}, Mean Loss {:.5f}".format(i, step, np.mean(total_loss)))
          step += 1
      except tf.errors.OutOfRangeError:
        sess.run(iterator.initializer) # reinitialize dataset.
    print("Start Testing...")
    total_test_acc = []
    sess.run(test_iterator.initializer)
    try:
      while True:
        ret = sess.run(test_acc)
        total_test_acc.append(ret)
    except tf.errors.OutOfRangeError:
      print("Finished.")
    print('Test Accuracy is: {:.4f}'.format(np.mean(total_test_acc)))
  g.close()

if __name__ == "__main__":
  cur_path = sys.path[0]
  argparser = argparse.ArgumentParser("Train TGN.")
  argparser.add_argument('--data_folder', type=str,
                         default=os.path.join(cur_path, '../../data/jodie/'),
                         help="Source dara folder, list files are node_table, edge_table, "
                              "train_table, val_table and test_table")
  argparser.add_argument('--files', type=str,
                         default="src_feat,dst_feat,wikipedia,wikipedia_train,wikipedia_val,wikipedia_test",
                         help="files names, join with `,`, in order of node_table, edge_table, "
                              "train_table, val_table and test_table")
  argparser.add_argument('--features_num', type=int, default=172)
  argparser.add_argument('--src_num', type=int, default=8226)
  argparser.add_argument('--dst_num', type=int, default=1000)
  argparser.add_argument('--time_dim', type=int, default=172)
  argparser.add_argument('--hidden_dim', type=int, default=128)
  argparser.add_argument('--num_heads', type=int, default=2)
  argparser.add_argument('--batch_size', type=int, default=200)
  argparser.add_argument('--nbr_size', nargs='+', type=int, default=[20, 10])
  argparser.add_argument('--dropout', type=float, default=0.1)
  argparser.add_argument('--epoch', type=int, default=50)
  argparser.add_argument('--lr', type=float, default=0.0001)

  args = argparser.parse_args()
  run(args)
