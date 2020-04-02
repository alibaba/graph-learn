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

import numpy as np
import graphlearn as gl

from trans_e import TransE

TRACKER_PATH  = './tracker/'
os.system('mkdir -p %s' % TRACKER_PATH)
os.system('rm -rf %s*' % TRACKER_PATH)

def load_graph():
  g = gl.Graph()\
        .node("../../data/FB15k-237/entity_node_table", node_type="entity",
              decoder=gl.Decoder(attr_types=["int"]))\
        .node("../../data/FB15k-237/relation_node_table", node_type="relation",
              decoder=gl.Decoder(attr_types=["int"]))\
        .edge("../../data/FB15k-237/train_tuple_table",
              edge_type=("entity", "entity", "hrt"),
              decoder=gl.Decoder(attr_types=["int"], weighted=False))
  return g

def train(config, graph):
  def model_fn():
    return TransE(graph,
                  config['neg_num'],
                  config['batch_size'],
                  config['margin'],
                  config['entity_num'],
                  config['relation_num'],
                  config['hidden_dim'],
                  s2h=config['s2h'],
                  ps_hosts=config['ps_hosts'])

  trainer = gl.LocalTFTrainer(model_fn,
                              epoch=config['epoch'],
                              optimizer=gl.get_tf_optimizer(
                              config['learning_algo'],
                              config['learning_rate']))
  trainer.train()
  entity_embs = trainer.get_node_embedding('entity')
  relation_embs = trainer.get_node_embedding('relation')
  return entity_embs, relation_embs


def main():
  config = {'epoch': 30,
            'batch_size': 3000,
            'ps_hosts': None,
            'neg_num': 25,
            'hidden_dim': 128,
            'out_dim': 128,
            'learning_algo': "adam",
            'learning_rate': 0.001,
            's2h': False,
            'margin': 1,
            'entity_num': 14541,
            'relation_num': 237,
            'emb_save_dir': './'}

  g = load_graph()
  g.init(server_id=0, server_count=1, tracker=TRACKER_PATH)

  e_embs, r_embs = train(config, g)
  print('begin dump embedding', config['emb_save_dir'])
  print(e_embs.shape, r_embs.shape)
  np.save(config['emb_save_dir'] + 'id_entity', e_embs)
  np.save(config['emb_save_dir'] + 'id_relation', r_embs)

if __name__ == "__main__":
  main()
