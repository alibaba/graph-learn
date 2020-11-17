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
import base64
import json
import sys
import numpy as np
import graphlearn as gl

from graph_sage import GraphSage

TRACKER_PATH  = './tracker/'
os.system('mkdir -p %s' % TRACKER_PATH)
os.system('rm -rf %s*' % TRACKER_PATH)


def train(config, graph):
  print('start training....')
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
                     agg_type=config['agg_type'],
                     node_type=config['node_type'],
                     edge_type=config['edge_type'],
                     train_node_type=config['node_type'])
  trainer = gl.LocalTFTrainer(model_fn,
                              epoch=config['epoch'],
                              optimizer=gl.get_tf_optimizer(
                                  config['learning_algo'],
                                  config['learning_rate'],
                                  config['weight_decay']))

  trainer.train()
  embs = trainer.get_node_embedding()
  # print(embs.shape)
  np.save(config['emb_save_dir'], embs)

from sklearn.metrics.pairwise import cosine_similarity
def test(config, graph):

  sampler_node = graph.node_sampler(config['node_type'], 1)
  nodes_set = set(sampler_node.get().ids)
  for i in range(10):
      nodes_list = sampler_node.get().ids
      nodes_set_tmp = set(nodes_list)
      nodes_set = nodes_set | nodes_set_tmp

  id_node = list(sorted(nodes_set))
  node_id = {}
  for i in range(len(id_node)):
      node_id[id_node[i]] = i

  test_count = 1.0
  find_count = 0.0
  embs = np.load(config['emb_save_dir'] + ".npy")
  sampler_edge = graph.edge_sampler(config['edge_type'], 1000)
  edges_list = sampler_edge.get()
  src_ids = edges_list.src_ids
  dst_ids = edges_list.dst_ids
  for j in range(len(src_ids)):
      if src_ids[j] in node_id and dst_ids[j] in node_id:
          vec_a = embs[node_id[src_ids[j]], :]
          vec_b = embs[node_id[dst_ids[j]], :]
          num = float(np.sum(vec_a * vec_b))
          denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
          cos = num / denom
          cos_sim = 0.5 + 0.5 * cos
          test_count = test_count + 1
          if cos_sim >= 0.1:
              find_count = find_count + 1

  acc = find_count/test_count
  print("Evaluation Results:")
  print("Predicted Edges in Dataset: %d / %d; Precision: %4f" %(find_count, test_count, acc))


def main():
  handle = sys.argv[1]
  task_index = 0
  task_count = 1
  s = base64.b64decode(handle).decode('utf-8')
  obj = json.loads(s)
  node_type = obj['node_schema'][0].split(':')[0]
  edge_type = obj['edge_schema'][0].split(':')[1]

  config = {'class_num': 32,
            'features_num': 2,
            'batch_size': 10, # 10
            'categorical_attrs_desc': '',
            'hidden_dim': 256,
            'in_drop_rate': 0.5,
            'hops_num': 2,
            'neighs_num': [10, 20],
            'full_graph_mode': False,
            'agg_type': 'gcn',  # mean, sum
            'learning_algo': 'adam',
            'learning_rate': 0.005,
            'weight_decay': 0.0005,
            'epoch': 1,
            'unsupervised': True,
            'use_neg': True,
            'neg_num': 10,
            'emb_save_dir': './id_emb',
            'node_type': node_type,
            'edge_type': edge_type}


  gl.set_vineyard_graph_id(obj['vineyard_id'])
  gl.set_vineyard_ipc_socket(obj['vineyard_socket'])
  gl.set_storage_mode(8)
  g = gl.get_graph_from_handle(handle, worker_index=task_index, worker_count=task_count, standalone=True)

  train(config, g)

  test(config, g)


if __name__ == "__main__":
  main()
