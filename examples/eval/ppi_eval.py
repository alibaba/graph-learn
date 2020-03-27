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
"""Evaluates F1 score for ppi dataset.

Reference:
  GraphSage Author's code https://github.com/williamleif/GraphSAGE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from argparse import ArgumentParser
import numpy as np
from networkx.readwrite import json_graph
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier


def train_and_test(train_embs, train_labels, test_embs, test_labels):
  np.random.seed(1)
  log = MultiOutputClassifier(SGDClassifier(loss="log"), n_jobs=10)
  log.fit(train_embs, train_labels)

  total = 0.0
  for i in range(test_labels.shape[1]):
    f1 = f1_score(test_labels[:, i],
                  log.predict(test_embs)[:, i],
                  average="macro")
    print("macro F1 score", f1)
    total += f1
  print("avg macro F1 score", total / test_labels.shape[1])


def main():
  parser = ArgumentParser("Caculates F1 score on PPI data.")
  parser.add_argument("--dataset_dir", default='../data/ppi',
                      help="Path to directory containing the dataset.")
  parser.add_argument("--emb_dir", default='../tf/graphsage',
                      help="Path of directory containing node embeddings.")
  parser.add_argument("--mode", default='test',
                      help="Evaluation mode, either val or test.")

  args = parser.parse_args()
  dataset_dir = args.dataset_dir
  embedding_file = args.emb_dir + '/id_emb.npy'
  mode = args.mode

  print("Loading...")
  G = json_graph.node_link_graph(json.load(open(dataset_dir + "/ppi-G.json")))
  labels = json.load(open(dataset_dir + "/ppi-class_map.json"))
  labels = {int(i): l for i, l in labels.iteritems()}

  train_ids = [n for n in G.nodes() if not G.node[n]['val']
               and not G.node[n]['test']]
  test_ids = [n for n in G.nodes() if G.node[n][mode]]
  train_labels = np.array([labels[i] for i in train_ids])
  if train_labels.ndim == 1:
    train_labels = np.expand_dims(train_labels, 1)
  test_labels = np.array([labels[i] for i in test_ids])

  embs = np.load(embedding_file)
  id_map = {}
  for i in range(embs.shape[0]):
    id_map[embs[i][0]] = i
  train_embs = embs[[id_map[k] for k in train_ids]][:, 1:]
  test_embs = embs[[id_map[k] for k in test_ids]][:, 1:]

  print("Running...")
  train_and_test(train_embs, train_labels, test_embs, test_labels)

if __name__ == "__main__":
  main()
