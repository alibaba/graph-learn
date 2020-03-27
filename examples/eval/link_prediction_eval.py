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
"""Evaluates AUC for link prediction."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
import numpy as np
from sklearn import metrics


def metric(emb_a, emb_b):
  emb_a = np.array(emb_a, dtype=np.float32)
  emb_b = np.array(emb_b, dtype=np.float32)

  return 1.0 / (1.0 + np.exp(-1 * np.sum(emb_a*emb_b)))

def fake_distribution_table(nodes_degree, train_ids):
  dist_table = []
  for idx in range(np.shape(nodes_degree)[0]):
    if int(nodes_degree[idx, 0]) in train_ids:
      dist_table.extend([nodes_degree[idx, 0]] * nodes_degree[idx, 1])

  return dist_table

def run_auc_homo(id_map, embs, nodes_degree, test_ids, train_ids, times=7):
  true_logits = []
  true_labels = []
  for idx in range(np.shape(test_ids)[0]):
    if int(test_ids[idx, 0]) not in train_ids \
      or int(test_ids[idx, 1]) not in train_ids:
      continue
    emb_a = embs[id_map[test_ids[idx, 0]], :]
    emb_b = embs[id_map[test_ids[idx, 1]], :]
    logit = metric(emb_a, emb_b)
    true_logits.append(logit)
    true_labels.append(1.0)

  dist_table = fake_distribution_table(nodes_degree, train_ids)
  neg_logits = []
  neg_labels = []
  for idx in range(np.shape(test_ids)[0]):
    if int(test_ids[idx, 0]) not in train_ids:
      continue
    for _ in range(times):
      random_idx = np.random.randint(0, len(dist_table))
      neg_id = dist_table[random_idx]
      emb_a = embs[id_map[test_ids[idx, 0]], :]
      emb_b = embs[id_map[neg_id], :]
      logit = metric(emb_a, emb_b)
      neg_logits.append(logit)
      neg_labels.append(0.0)

  logits = np.concatenate([true_logits, neg_logits], axis=0)
  labels = np.concatenate([true_labels, neg_labels], axis=0)
  auc = metrics.roc_auc_score(labels, logits)

  print("auc:", auc)

def run_auc_bi(u_id_map, u_embs,
               i_id_map, i_embs,
               nodes_degree, test_ids,
               u_train_ids, i_train_ids,
               times=7):
  #pos logits
  true_logits = []
  true_labels = []
  for idx in range(np.shape(test_ids)[0]):
    if int(test_ids[idx, 0]) not in u_train_ids \
      or int(test_ids[idx, 1]) not in i_train_ids:
      continue
    emb_a = u_embs[u_id_map[test_ids[idx, 0]], :]
    emb_b = i_embs[i_id_map[test_ids[idx, 1]], :]
    logit = metric(emb_a, emb_b)
    true_logits.append(logit)
    true_labels.append(1.0)

  #fake distribution
  dist_table = fake_distribution_table(nodes_degree, i_train_ids)

  neg_logits = []
  neg_labels = []
  for idx in range(np.shape(test_ids)[0]):
    if int(test_ids[idx, 0]) not in u_train_ids:
      continue
    for _ in range(times):
      random_idx = np.random.randint(0, len(dist_table))
      neg_id = dist_table[random_idx]
      emb_a = u_embs[u_id_map[test_ids[idx, 0]], :]
      emb_b = i_embs[i_id_map[neg_id], :]
      logit = metric(emb_a, emb_b)
      neg_logits.append(logit)
      neg_labels.append(0.0)

  logits = np.concatenate([true_logits, neg_logits], axis=0)
  labels = np.concatenate([true_labels, neg_labels], axis=0)
  auc = metrics.roc_auc_score(labels, logits)

  print("auc:", auc)


def read_embedding(file_name):
  embs = np.load(file_name)
  print(embs.shape)
  id_map = {}
  for i in range(embs.shape[0]):
    id_map[int(embs[i][0])] = i

  embs = embs[:, 1:]

  return id_map, embs

if __name__ == '__main__':
  parser = ArgumentParser("Calcualtes AUC for link-prediction")
  parser.add_argument("--dataset_dir", default='../data/arxiv',
                      help="Path to directory containing the dataset.")
  parser.add_argument("--emb_dir", default='../tf/line',
                      help="Path of node embeddings.")
  parser.add_argument("--graph_type", default='homo',
                      help="graph_type homo or u2i")
  args = parser.parse_args()

  dataset_dir = args.dataset_dir
  emb_dir = args.emb_dir
  emb_file = emb_dir + '/id_emb.npy'
  u_emb_file = emb_dir + '/u_emb.npy'
  i_emb_file = emb_dir + '/i_emb.npy'
  graph_type = args.graph_type

  test_ids = np.load(dataset_dir + '/test_ids.npy')
  nodes_degree = np.load(dataset_dir + "/id_degree.npy")
  if graph_type == 'homo':
    train_ids = set()
    print("Loading...")
    with open(dataset_dir + "/arxiv-links-train-edge", 'r') as f:
      lines = f.readlines()
      for idx in range(1, len(lines)):
        lineSplit = lines[idx].strip().split('\t')
        train_ids.add(int(lineSplit[0]))
        train_ids.add(int(lineSplit[1]))
    id_map, embs = read_embedding(emb_file)

    print("Running...")
    run_auc_homo(id_map, embs, nodes_degree, test_ids, train_ids)
  else:
    u_train_ids = set()
    i_train_ids = set()
    print("Loading...")
    with open(dataset_dir + "/u2i_20200222_train", 'r') as f:
      lines = f.readlines()
      for idx in range(1, len(lines)):
        lineSplit = lines[idx].strip().split('\t')
        u_train_ids.add(int(lineSplit[0]))
        i_train_ids.add(int(lineSplit[1]))

    u_id_map, u_embs = read_embedding(u_emb_file)
    i_id_map, i_embs = read_embedding(i_emb_file)

    print("Running...")
    run_auc_bi(u_id_map, u_embs, i_id_map, i_embs,
               nodes_degree, test_ids, u_train_ids, i_train_ids)
