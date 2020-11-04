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
import tensorflow as tf

from graph_sage import GraphSage


def train_local(config, graph):
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
                     train_node_type=config['train_node_type'])
  trainer = gl.LocalTFTrainer(model_fn,
                              epoch=config['epoch'],
                              optimizer=gl.get_tf_optimizer(
                                  config['learning_algo'],
                                  config['learning_rate'],
                                  config['weight_decay']))
  print("start to train...")
  trainer.train()
  emb_ids, emb_values = trainer.get_node_embedding_fixed()
  print('shape: ', emb_ids.shape, emb_values.shape)
  np.save(config['emb_save_dir'] + '_ids', emb_ids)
  np.save(config['emb_save_dir'] + '_values', emb_values)


from sklearn.metrics.pairwise import cosine_similarity
def test_local(config, graph):
  test_count = 0.0
  find_count = 0.0
  emb_ids = np.load(config['emb_save_dir'] + "_ids.npy")
  emb_values = np.load(config['emb_save_dir'] + "_values.npy")
  id_map = {}
  for idx, vid in enumerate(emb_ids):
    id_map[vid] = idx
  sampler_edge = graph.edge_sampler(config['edge_type'], 1000)
  edges_list = sampler_edge.get()
  src_ids = edges_list.src_ids
  dst_ids = edges_list.dst_ids
  for j in range(len(src_ids)):
    vec_a = emb_values[id_map[src_ids[j]]]
    vec_b = emb_values[id_map[dst_ids[j]]]
    num = float(np.sum(vec_a * vec_b))
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
      cos = 0
    else:
      cos = num / denom
      cos_sim = 0.5 + 0.5 * cos
      test_count = test_count + 1
    if cos_sim >= 0.7:
      find_count = find_count + 1

  acc = find_count/test_count
  print("Evaluation Results:")
  print("Predicted Edges in Dataset: %d / %d; Precision: %4f" %(find_count, test_count, acc))


def train_dist(config, graph):
  def model_fn():
    return GraphSage(graph,
                     config['class_num'],
                     config['features_num'],
                     config['batch_size'],
                     val_batch_size=config['val_batch_size'],
                     test_batch_size=config['test_batch_size'],
                     categorical_attrs_desc=config['categorical_attrs_desc'],
                     hidden_dim=config['hidden_dim'],
                     in_drop_rate=config['in_drop_rate'],
                     neighs_num=config['neighs_num'],
                     full_graph_mode=config['full_graph_mode'],
                     unsupervised=config['unsupervised'],
                     agg_type=config['agg_type'],
                     node_type=config['node_type'],
                     edge_type=config['edge_type'],
                     train_node_type=config['node_type'])

  cluster = tf.train.ClusterSpec(
    {'ps': config['ps_hosts'], 'worker': config['worker_hosts']}
  )
  trainer = gl.DistTFTrainer(model_fn,
                             cluster_spec=cluster,
                             task_name=config['job_name'],
                             task_index=config['task_index'],
                             epoch=config['epoch'],
                             optimizer=gl.get_tf_optimizer(
                                 config['learning_algo'],
                                 config['learning_rate'],
                                 config['weight_decay']))
  task_index = config['task_index']
  if config['job_name'] == 'worker': # also graph-learn client in this example.
    trainer.train()
    embs = trainer.get_node_embedding()
    np.save(config['emb_save_dir'] + str(task_index), embs)
    print("embds shape:", embs.shape)
  else:
    trainer.join()


def local_training(handle, config):
  features = ["creationDate"]
  g = gl.Graph().vineyard(handle, nodes=["person"], edges=[("person", "knows", "person")]) \
      .node_attributes("person", features, n_int=1, n_float=0, n_string=0) \
      .init_vineyard(standalone=True)
  train_local(config, g)
  test_local(config, g)
  g.close()


def dist_training(handle, config):
  hosts = handle['hosts'].split(',')
  # use the first half as worker, others as server
  # NOTE: gl_hosts and ps_hosts must has diff endpoint if tracker mode is 0.
  mid = len(hosts) // 2
  config['worker_hosts'] = [
    "{}:{}".format(pod_name, 7000 + index) for index, pod_name in enumerate(hosts[0:mid])
  ]
  config['ps_hosts'] = [
    "{}:{}".format(pod_name, 8000 + index) for index, pod_name in enumerate(hosts[mid:])
  ]
  config['gl_hosts'] = [
    "{}:{}".format(pod_name, 9000 + index) for index, pod_name in enumerate(hosts[mid:])
  ]
  handle['server'] = ','.join(config['gl_hosts'])
  handle['client'] = ','.join(config['worker_hosts'])

  # check is worker or ps
  pod_index = handle['pod_index']
  features = ['creationDate']
  if pod_index < mid:
    config['job_name'] = 'worker'
    config['task_index'] = pod_index
    client_count = len(config['worker_hosts'])
    g = gl.Graph().vineyard(handle, nodes=["person"], edges=[("person", "knows", "person")]) \
        .node_attributes("person", features, n_int=1, n_float=0, n_string=0) \
        .init_vineyard(worker_index=config['task_index'], worker_count=client_count)
  else:
    config['job_name'] = 'ps'
    config['task_index'] = pod_index - mid
    g = gl.Graph().vineyard(handle, nodes=["person"], edges=[("person", "knows", "person")]) \
        .node_attributes("person", features, n_int=1, n_float=0, n_string=0) \
        .init_vineyard(server_index=config['task_index'], worker_count=client_count)

  # training
  train_dist(config, g)
  g.close()


def main():
  handle_str = sys.argv[1]
  s = base64.b64decode(handle_str).decode('utf-8')
  handle = json.loads(s)
  handle['pod_index'] = int(sys.argv[2])

  config = {'class_num': 32,
            'features_num': 2,
            'batch_size': 10, # 10
            'val_batch_size': 10,
            'test_batch_size': 10,
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
            'node_type': "person",
            'edge_type': "knows",
            'train_node_type': "person"}

  host_num = len(handle['hosts'].split(','))
  if host_num  == 1:
    local_training(handle, config)
  elif host_num > 1:
    dist_training(handle, config)
  else:
    raise ValueError('Not found hosts in handle.')


if __name__ == "__main__":
  main()
