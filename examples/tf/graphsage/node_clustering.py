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

import base64
import json
import os
import sys

import numpy as np
import graphlearn as gl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from graph_sage import GraphSage


def train(config, graph, n_clusters=7, seed=0):
  """
  Run unsupervised GraphSage and extract node embeddings.
  Then, run Kmeans cluster to cluster the nodes.
  n_cluster: set to the number of categories of actual node type
  """
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
  print("start to train...")
  trainer.train()
  ids, embs = trainer.get_node_embedding_fixed()
  print('shape: ', ids.shape, embs.shape)

  data = StandardScaler().fit_transform(embs)
  pca = PCA(n_components=4)
  principalComponents = pca.fit_transform(data)
  kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
  clusters = kmeans.fit_predict(principalComponents)

  return ids, clusters


def test(config, node_ids, clusters, num_sample=10000):
  """
  evaluate the unsupervised clustering.
  node_ids should be of type list
  """
  f = open("id_label.csv", "r")
  ground_truth = dict()
  for l in f.readlines():
    l = l.split("\t")
    ground_truth[l[0]] = l[1]
  f.close()

  predicted = dict()
  for i in range(len(node_ids)):
    predicted[node_ids[i]] = clusters[i]

  #randomly generate pairs of ids for testing
  test_id1 = np.random.choice(node_ids, size=num_sample, replace=True)
  test_id2 = np.random.choice(node_ids, size=num_sample, replace=True)

  total_num = 0
  correct_num = 0
  for i in range(num_sample):
    if test_id1[i] == test_id2[i]:
      continue
    cl1 = predicted[test_id1[i]]
    cl2 = predicted[test_id2[i]]
    true1 = ground_truth[test_id1[i]]
    true2 = ground_truth[test_id2[i]]
    if (cl1 == cl2 and true1 == true2) or (cl1 != cl2 and true1 != true2):
      correct_num += 1
    total_num += 1

  print(f"Clustering accuracy is approaximately: {correct_num/total_num}")
  return correct_num/total_num


def main():
  handle_str = sys.argv[1]
  s = base64.b64decode(handle_str).decode('utf-8')
  handle = json.loads(s)

  config = {'class_num': 16, # output dimension
            'features_num': 130, # 128 dimension + kcore + page_rank
            'batch_size': 1000,
            'categorical_attrs_desc': '',
            'hidden_dim': 256,
            'in_drop_rate': 0.5,
            'hops_num': 2,
            'neighs_num': [5, 5],  # [1, 1] to make it fase, origin is [10, 20]
            'full_graph_mode': False,
            'agg_type': 'gcn',  # mean, sum
            'learning_algo': 'adam',
            'learning_rate': 0.005,
            'weight_decay': 0.0005,
            'epoch': 1, # 1 to make it fase in ci, origin is 20
            'unsupervised': True,
            'use_neg': True,
            'neg_num': 10,
            'node_type': "paper",
            'edge_type': "cites"}

  features = []
  for i in range(128):
    features.append("feat_" + str(i))
  features.append("KC")
  features.append("TC")
  try:
    g = gl.Graph().vineyard(handle, nodes=["paper"], edges=[("paper", "cites", "paper")]) \
        .node_attributes("paper", features, n_int=2, n_float=128, n_string=0) \
        .init_vineyard(standalone=True)

    node_ids, clusters = train(config, g, n_clusters=40)  # 40 for test
    # TODO: uncomment after we use oid as output ids
    # test(config, node_ids, clusters)
    g.close()
  except Exception as e:
    g.close()
    raise RuntimeError() from e
  except KeyboardInterrupt:
    g.close()


if __name__ == "__main__":
  main()
