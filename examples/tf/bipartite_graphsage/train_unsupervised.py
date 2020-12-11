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

import os

import graphlearn as gl
import numpy as np
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

from bipartite_graph_sage import BipartiteGraphSage

TRACKER_PATH  = './tracker/'
os.system('mkdir -p %s' % TRACKER_PATH)
os.system('rm -rf %s*' % TRACKER_PATH)

def train(config, graph):
  def model_fn():
    return  BipartiteGraphSage(graph,
                               config['batch_size'],
                               config['hidden_dim'],
                               config['output_dim'],
                               config['hops_num'],
                               config['u_neighs_num'],
                               config['i_neighs_num'],
                               u_features_num=config['u_features_num'],
                               u_categorical_attrs_desc=config['u_categorical_attrs_desc'],
                               i_features_num=config['i_features_num'],
                               i_categorical_attrs_desc=config['i_categorical_attrs_desc'],
                               neg_num=config['neg_num'],
                               use_input_bn=config['use_input_bn'],
                               act=config['act'],
                               agg_type=config['agg_type'],
                               need_dense=config['need_dense'],
                               in_drop_rate=config['drop_out'],
                               ps_hosts=config['ps_hosts'])
  trainer = gl.LocalTFTrainer(model_fn,
                              epoch=config['epoch'],
                              optimizer=gl.get_tf_optimizer(
                                  config['learning_algo'],
                                  config['learning_rate'],
                                  config['weight_decay']))

  trainer.train()

  u_embs = trainer.get_node_embedding("u")
  np.save('u_emb', u_embs)

  i_embs = trainer.get_node_embedding("i")
  np.save('i_emb', i_embs)

def load_graph(config):
  g = gl.Graph()\
        .node("../../data/u2i/u2i_node_attrs", node_type="i",
              decoder=gl.Decoder(attr_types=["string"]))\
        .node("../../data/u2i/u2i_node_attrs", node_type="u",
              decoder=gl.Decoder(attr_types=["string"]))\
        .edge("../../data/u2i/u2i_20200222_train", edge_type=("u", "i", "u-i"),
              decoder=gl.Decoder(weighted=True), directed=False)
  return g


def main():
  config = {'batch_size': 128,
            'hidden_dim': 128,
            'output_dim': 128,
            'u_features_num': 1,
            'u_categorical_attrs_desc': {"0":["u_id",10000,128]},
            'i_features_num': 1,
            'i_categorical_attrs_desc': {"0":["i_id",10000,128]},
            'hops_num': 1,
            'u_neighs_num': [10],
            'i_neighs_num': [10],
            'neg_num': 10,
            'learning_algo': 'adam',
            'learning_rate': 0.001,
            'weight_decay': 0.0005,
            'epoch': 10,
            'use_input_bn': True,
            'act': tf.nn.leaky_relu,
            'agg_type': 'gcn',
            'need_dense': True,
            'drop_out': 0.0,
            'ps_hosts': None}

  g = load_graph(config)
  g.init(server_id=0, server_count=1, tracker=TRACKER_PATH)
  train(config, g)

if __name__ == "__main__":
  main()
