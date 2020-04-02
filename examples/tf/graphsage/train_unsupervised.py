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
"""local training script for unsupervised GraphSage"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import graphlearn as gl

from graph_sage import GraphSage

TRACKER_PATH  = './tracker/'
os.system('mkdir -p %s' % TRACKER_PATH)
os.system('rm -rf %s*' % TRACKER_PATH)


def load_graph(config):
  dataset_folder = config['dataset_folder']
  node_type = config['node_type']
  edge_type = config['edge_type']
  g = gl.Graph()\
        .node(dataset_folder + "node_table", node_type=node_type,
              decoder=gl.Decoder(attr_types=["float"]*50))\
        .edge(dataset_folder + "edge_table",
              edge_type=(node_type, node_type, edge_type),
              decoder=gl.Decoder(weighted=True), directed=False)\
        .node(dataset_folder + "node_table", node_type="train",
              decoder=gl.Decoder(attr_types=["float"]*50))
  return g

def train(config, graph):
  def model_fn():
    return GraphSage(graph,
                     config['class_num'],
                     config['features_num'],
                     config['batch_size'],
                     categorical_attrs_desc=config['categorical_attrs_desc'],
                     hidden_dim=config['hidden_dim'],
                     in_drop_rate=config['in_drop_rate'],
                     neighs_num=config['neighs_num'],
                     full_graph_mode=config['full_graph_mode'],
                     unsupervised=config['unsupervised'],
                     neg_num=config['neg_num'],
                     agg_type=config['agg_type'])
  trainer = gl.LocalTFTrainer(model_fn,
                              epoch=config['epoch'],
                              optimizer=gl.get_tf_optimizer(
                                  config['learning_algo'],
                                  config['learning_rate'],
                                  config['weight_decay']))
  trainer.train()
  embs = trainer.get_node_embedding()
  np.save(config['emb_save_dir'], embs)

def main():
  config = {'dataset_folder': '../../data/ppi/',
            'class_num': 128,
            'features_num': 50,
            'batch_size': 512, # 10
            'categorical_attrs_desc': '',
            'hidden_dim': 128,
            'in_drop_rate': 0.5,
            'hops_num': 2,
            'neighs_num': [25, 10],
            'full_graph_mode': False,
            'agg_type': 'gcn',  # mean, sum
            'learning_algo': 'adam',
            'learning_rate': 0.01,
            'weight_decay': 0.0005,
            'epoch': 1,
            'unsupervised': True,
            'use_neg': True,
            'neg_num': 20,
            'emb_save_dir': './id_emb',
            'node_type': 'item',
            'edge_type': 'relation'}

  g = load_graph(config)
  g.init(server_id=0, server_count=1, tracker=TRACKER_PATH)
  train(config, g)

if __name__ == "__main__":
  main()
