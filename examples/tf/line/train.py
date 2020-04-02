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

from line import LINE

TRACKER_PATH  = './tracker/'
os.system('mkdir -p %s' % TRACKER_PATH)
os.system('rm -rf %s*' % TRACKER_PATH)

def train(config, graph):
  def model_fn():
    return LINE(graph,
                config['node_count'],
                config['hidden_dim'],
                config['neg_num'],
                config['batch_size'],
                config['s2h'],
                config['ps_hosts'],
                config['proximity'],
                config['node_type'],
                config['edge_type'])
  trainer = gl.LocalTFTrainer(model_fn,
                              epoch=config['epoch'],
                              optimizer=gl.get_tf_optimizer(
                                  config['learning_algo'],
                                  config['learning_rate']))
  trainer.train()
  embs = trainer.get_node_embedding()
  np.save(config['emb_save_dir'], embs)

def load_graph(config):
  node_type = config['node_type']
  edge_type = config['edge_type']
  g = gl.Graph()\
        .node("../../data/arxiv/arxiv-links-train-node-attrs",
              node_type=node_type,
              decoder=gl.Decoder(attr_types=["int"])) \
        .edge("../../data/arxiv/arxiv-links-train-edge",
              edge_type=(node_type, node_type, edge_type),
              decoder=gl.Decoder(weighted=True), directed=False)

  return g

def main():
  config = {'node_count': 10000,
            'hidden_dim': 128,
            'batch_size': 32,
            'neg_num': 2,
            'epoch': 200,
            'learning_algo': 'sgd',
            'learning_rate': 10,
            'proximity': 'first_order',
            's2h': False,
            'emb_save_dir': "./id_emb",
            'ps_hosts': None,
            'node_type': 'item',
            'edge_type': 'relation'}

  g = load_graph(config)
  g.init(server_id=0, server_count=1, tracker=TRACKER_PATH)

  train(config, g)


if __name__ == "__main__":
  main()
