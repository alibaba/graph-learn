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
        .node(dataset_folder + "node_table_50", node_type=node_type,
              decoder=gl.Decoder(attr_types=["float", "string", "string", "string", "string", "string"]))\
        .edge(dataset_folder + "edge_table_50",
              edge_type=(node_type, node_type, edge_type),
              decoder=gl.Decoder(attr_types=["float"], weighted=False), directed=True)\
        .node(dataset_folder + "node_table_50", node_type="train",
              decoder=gl.Decoder(attr_types=["float", "string", "string", "string", "string", "string"]))
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

  # print(id_node)
  # print(node_id)

  test_count = 1.0
  find_count = 0.0
  embs = np.load(config['emb_save_dir'] + ".npy")
  sampler_edge = graph.edge_sampler(config['edge_type'], 1000)
  edges_list = sampler_edge.get()
  src_ids = edges_list.src_ids
  dst_ids = edges_list.dst_ids
  for j in range(len(src_ids)):
      if node_id.has_key(src_ids[j]) and node_id.has_key(dst_ids[j]):
          vec_a = embs[node_id[src_ids[j]], :]
          vec_b = embs[node_id[dst_ids[j]], :]
          num = float(np.sum(vec_a * vec_b))
          denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
          cos = num / denom
          cos_sim = 0.5 + 0.5 * cos
          test_count = test_count + 1
          if cos_sim >= 0.98:
              find_count = find_count + 1
  
  acc = find_count/test_count
  print("Evaluation Results:")
  print("Predicted Edges in Dataset: %d / %d; Precision: %4f" %(find_count, test_count, acc))

  '''
  node_list = {}
  id_list = {}
  read_n = open(config['dataset_folder'] + 'node_map', 'r')
  n_lines = read_n.readlines()
  for i in range(1, len(n_lines)):
      n_l  = n_lines[i].split('\t')
      node_list[int(n_l[0])] = int(n_l[1])
      id_list[int(n_l[1])] = int(n_l[0])
  read_n.close()
  node_num = len(node_list) - 1

  embs = np.load(config['emb_save_dir'] + ".npy")
  find_count = 0.0
  read_e = open(config['dataset_folder'] + 'edge_table_test', 'r')
  e_lines = read_e.readlines()
  edge_count = float(len(e_lines))
  for j in range(1, len(e_lines)):
      e_l = e_lines[j].split('\t')
      vec_a = embs[node_list[int(e_l[0])], :]
      vec_b = embs[node_list[int(e_l[1])], :]
      num = float(np.sum(vec_a * vec_b))
      denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
      cos = num / denom
      cos_sim = 0.5 + 0.5 * cos
      if cos_sim >= 0.98:
          find_count = find_count + 1.0
  read_e.close()

  nofind_count = 0.0
  noedge_count = 10000
  for i in range(noedge_count):
      n1 = np.random.randint(100, 9000)
      n2 = np.random.randint(100, 9000)
      vec_a = embs[n1, :]
      vec_b = embs[n2, :]
      num = float(np.sum(vec_a * vec_b))
      denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
      cos = num / denom
      cos_sim = 0.5 + 0.5 * cos
      if cos_sim <= 0.98:
          nofind_count = nofind_count + 1.0

  acc = (find_count + nofind_count) / (edge_count + noedge_count)
  #acc = 2 * find_count / (node_num * (node_num - 1))
  #fs = rec * acc / (rec + acc)
  print("Evaluation Results:")
  #print("Predicted Edges: %d / %d; Non-Predicted Edges: %d / %d" %(find_count, edge_count, nofind_count, noedge_count) )
  print("Accurately Predicted Edges in Testing Dataset: %d / %d" %(find_count + nofind_count, edge_count + noedge_count) )
  print("Precision: %2f" %(acc))
  '''

def main():
  config = {'dataset_folder': '../../data/ldbc_50/',
            'class_num': 32,
            'features_num': 1,
            'batch_size': 5, # 10
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
            'node_type': 'item',
            'edge_type': 'relation'}

  g = load_graph(config)
  g.init(server_id=0, server_count=1, tracker=TRACKER_PATH)
  train(config, g)
  test(config, g)


if __name__ == "__main__":
  main()
