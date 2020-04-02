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
"""Training script of DeepWalk"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import graphlearn as gl
import numpy as np

from deepwalk import DeepWalk

TRACKER_PATH  = './tracker/'
os.system('mkdir -p %s' % TRACKER_PATH)
os.system('rm -rf %s*' % TRACKER_PATH)

def load_graph(config):
  node_type = config['node_type']
  edge_type = config['edge_type']
  g = gl.Graph().edge("../../data/blogcatelog/edge_table",
                      edge_type=(node_type, node_type, edge_type),
                      decoder=gl.Decoder(weighted=True), directed=False)\
                .node("../../data/blogcatelog/node_table", node_type=node_type,
                      decoder=gl.Decoder(weighted=True))
  return g

def train(config, graph):
  def model_fn():
    return DeepWalk(graph,
                    config['walk_len'],
                    config['window_size'],
                    config['node_count'],
                    config['hidden_dim'],
                    config['neg_num'],
                    config['batch_size'],
                    s2h=config['s2h'],
                    ps_hosts=config['ps_hosts'],
                    temperature=config['temperature'])
  trainer = gl.LocalTFTrainer(model_fn,
                              epoch=config['epoch'],
                              optimizer=gl.get_tf_optimizer(
                                config['learning_algo'],
                                config['learning_rate']))
  trainer.train()
  embs = trainer.get_node_embedding()
  np.save(config['emb_save_dir'], embs)

def main():
  config = {'walk_len': 20,
            'window_size': 5,
            'node_count': 10312,
            'hidden_dim': 128,
            'batch_size': 128,
            'neg_num': 10,
            'epoch': 40,
            'learning_algo': 'adam',
            'learning_rate': 0.01,
            'emb_save_dir': "./id_emb",
            's2h': False,
            'ps_hosts': None,
            'temperature': 1.0,
            'node_type': 'item',
            'edge_type': 'relation'}

  g = load_graph(config)
  g.init(server_id=0, server_count=1, tracker=TRACKER_PATH)
  train(config, g)

if __name__ == "__main__":
  main()
