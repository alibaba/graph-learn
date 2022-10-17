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

from edge_cn_inducer import EdgeCNInducer

def load_graph(config):
  data_dir = config['dataset_folder']
  g = gl.Graph() \
    .node(data_dir+'ogbl_collab_node', node_type='i',
          decoder=gl.Decoder(attr_types=['float'] * config['features_num'],
                             attr_dims=[0]*config['features_num'])) \
    .edge(data_dir+'ogbl_collab_train_edge', edge_type=('i', 'i', 'train'),
          decoder=gl.Decoder(weighted=True), directed=False) \
    .edge(data_dir+'ogbl_collab_val_edge', edge_type=('i', 'i', 'val'),
          decoder=gl.Decoder(weighted=True), directed=True) \
    .edge(data_dir+'ogbl_collab_val_edge_neg', edge_type=('i', 'i', 'val_neg'),
          decoder=gl.Decoder(weighted=True), directed=True) \
    .edge(data_dir+'ogbl_collab_test_edge', edge_type=('i', 'i', 'test'),
          decoder=gl.Decoder(weighted=True), directed=True) \
    .edge(data_dir+'ogbl_collab_test_edge_neg', edge_type=('i', 'i', 'test_neg'),
          decoder=gl.Decoder(weighted=True), directed=True)
  return g

def eval_hits(y_pred_pos, y_pred_neg, k):
  '''
      compute Hits@K
      For each positive target node, the negative target nodes are the same.
      y_pred_neg is an array.
      rank y_pred_pos[i] against y_pred_neg for each i
  '''
  if len(y_pred_neg) < k or len(y_pred_pos) == 0:
    return {'hits@{}'.format(k): 1.}
  kth_score_in_negative_edges = np.sort(y_pred_neg)[-k]
  hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)
  return {'hits@{}'.format(k): hitsK}

def train(g, model, predictor, config):
  tfg.conf.training = True
  seed = g.E('train').batch(config['batch_size']).shuffle(traverse=True)
  src = seed.outV().alias('pos_src')
  src.outV('train').sample(config['nbrs_num']).by('full').alias('src_hop1')
  dst = seed.inV().alias('pos_dst')
  dst.outV('train').sample(config['nbrs_num']).by('full').alias('dst_hop1')
  src.outNeg('train').sample(1).by('random').alias('neg_dst').\
    outV('train').sample(config['nbrs_num']).by('full').alias('neg_hop1')
  query = seed.values()
  dataset = tfg.Dataset(query, inducer=EdgeCNInducer(
    use_neg=True, addl_types_and_shapes=config['strut_label_spec']))
  pos_graph, neg_graph = dataset.get_batchgraph()
  pos_src, pos_dst = model.forward(batchgraph=pos_graph)
  neg_src, neg_dst = model.forward(batchgraph=neg_graph)
  pos_h = predictor(pos_src * pos_dst)
  neg_h = predictor(neg_src * neg_dst)
  # train loss
  loss = tfg.sigmoid_cross_entropy_loss(pos_h, neg_h)
  return dataset.iterator, loss

def test(g, model, predictor, config, edge_type='test'):
  tfg.conf.training = False
  seed = g.E(edge_type).batch(config['batch_size'])
  src = seed.outV().alias('pos_src')
  src.outV('train').sample(config['nbrs_num']).by('full').alias('src_hop1')
  dst = seed.inV().alias('pos_dst')
  dst.outV('train').sample(config['nbrs_num']).by('full').alias('dst_hop1')
  query = seed.values()
  dataset = tfg.Dataset(query, inducer=EdgeCNInducer(
    addl_types_and_shapes=config['strut_label_spec']))
  pos_graph, _ = dataset.get_batchgraph()
  pos_src, pos_dst = model.forward(batchgraph=pos_graph)
  logits = predictor(pos_src * pos_dst)
  return dataset.iterator, logits

def run(config):
  # graph input data
  g = load_graph(config=config)
  g.init()
  # model
  model = tfg.SEAL(config['batch_size'],
                   input_dim=config['features_num'],
                   hidden_dim=config['hidden_dim'],
                   output_dim=config['output_dim'],
                   depth=config['depth'],
                   drop_rate=config['drop_out'],
                   agg_type=config['agg_type'])
  predictor = tfg.LinkPredictor(name="link_pred", 
    input_dim=config['output_dim'], num_layers=config['predictor_layers'])
  def eval(sess, logits, iterator):
    """evaluate accuracy"""
    sess.run(iterator.initializer)
    outs = np.array([])
    while True:
      try:
        outs = np.append(sess.run(logits), outs)
      except tf.errors.OutOfRangeError:
        break
    return outs

  # trainer
  iterator, loss = train(g, model, predictor, config)
  optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate'])
  train_op = optimizer.minimize(loss)
  train_ops = [loss, train_op]
  test_iter, test_logits = test(g, model, predictor, config, edge_type='test')
  test_neg_iter, test_neg_logits = test(g, model, predictor, config, edge_type='test_neg')
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    epoch = 0
    for step in range(config['steps']):
      try:
        ret = sess.run(train_ops)
        if step and step % 100 == 0:
          print(datetime.datetime.now(),
                'Epoch {}, Iter {}, Loss {:.5f}'.format(epoch, step, ret[0]))
        if step % 1000 == 0: # test
          pos_logits = eval(sess, test_logits, test_iter)
          neg_logits = eval(sess, test_neg_logits, test_neg_iter)
          print('Test hits@50:', eval_hits(pos_logits, neg_logits, 50))
      except tf.errors.OutOfRangeError:
        sess.run(iterator.initializer) # reinitialize dataset.
        epoch += 1
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
            'predictor_layers': 3,
            'steps': 100000,
            'attn_heads': 8, # used for GAT.
            'strut_label_spec': {'struct_label': [tf.int32, tf.TensorShape([None])]}
           }
  run(config)