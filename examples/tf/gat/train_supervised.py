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
"""Training script for GAT algorithm"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import graphlearn as gl
from gat import GAT

TRACKER_PATH  = './tracker/'
os.system('mkdir -p %s' % TRACKER_PATH)
os.system('rm -rf %s*' % TRACKER_PATH)

def load_graph(config):
  """
  load graph from source, decode
  :return:
  """
  dataset_folder = config['dataset_folder']
  node_type = config['node_type']
  edge_type = config['edge_type']
  g = gl.Graph()\
        .node(dataset_folder + "node_table", node_type=node_type,
              decoder=gl.Decoder(labeled=True,
                                 attr_types=["float"] *
                                            (config['features_num']),
                                 attr_delimiter=":"))\
        .edge(dataset_folder + "edge_table_with_self_loop",
              edge_type=(node_type, node_type, edge_type),
              decoder=gl.Decoder(weighted=True), directed=False)\
        .node(dataset_folder + "train_table", node_type="train",
              decoder=gl.Decoder(weighted=True))\
        .node(dataset_folder + "val_table", node_type="val",
              decoder=gl.Decoder(weighted=True))\
        .node(dataset_folder + "test_table", node_type="test",
              decoder=gl.Decoder(weighted=True))
  return g


def train(config, graph):
  def model_fn():
    return GAT(graph,
               config['class_num'],
               config['features_num'],
               config['num_heads'],
               config['batch_size'],
               val_batch_size=config['val_batch_size'],
               test_batch_size=config['test_batch_size'],
               categorical_attrs_desc=config['categorical_attrs_desc'],
               hidden_dim=config['hidden_dim'],
               in_drop_rate=config['in_drop_rate'],
               attn_drop_rate=config['attn_drop_rate'],
               neighs_num=config['neighs_num'],
               hops_num=config['hops_num'],
               full_graph_mode=config['full_graph_mode'])

  trainer = gl.LocalTFTrainer(model_fn,
                              epoch=config['epoch'],
                              optimizer=gl.get_tf_optimizer(
                                  config['learning_algo'],
                                  config['learning_rate'],
                                  config['weight_decay']))
  trainer.train_and_evaluate()


def main():
  config = {'dataset_folder': '../../data/cora/',
            'class_num': 7,
            'features_num': 1433,
            'batch_size': 140,
            'val_batch_size': 300,
            'test_batch_size': 1000,
            'categorical_attrs_desc': '',
            'hidden_dim': 16,
            'num_heads': [8, 1],
            'in_drop_rate': 0.6,
            'attn_drop_rate': 0.6,
            'hops_num': 2,
            'neighs_num': None, #[5, 2],
            'full_graph_mode': True,
            'learning_algo': 'adamW',
            'learning_rate': 0.005,
            'weight_decay': 0.0005,
            'epoch': 200,
            'node_type': 'item',
            'edge_type': 'relation'}

  g = load_graph(config)
  g.init(server_id=0, server_count=1, tracker=TRACKER_PATH)
  train(config, g)


if __name__ == "__main__":
  main()
