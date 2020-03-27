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
"""Evaluates F1 score using multi-label classification for BlogCateLogData,
used by DeepWalk & Node2Vec.

Reference:
  DeepWalk Author's code https://github.com/phanein/deepwalk.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
from six import iteritems
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle as skshuffle

NODE_NUM = 10312
CLASS_NUM = 39


class MultiLabelClassifier(OneVsRestClassifier):
  def predict_topk(self, emb, topk_list):
    """return predict topk result."""
    assert emb.shape[0] == len(topk_list)
    probs = np.asarray(super(MultiLabelClassifier, self).predict_proba(emb))
    topk_probs = []
    for idx, k in enumerate(topk_list):
      prob = probs[idx, :]
      topk_prob = self.classes_[prob.argsort()[-k:]].tolist()
      topk_probs.append(topk_prob)
    return topk_probs


def train_and_test(repeat_data):
  all_ratio_results = defaultdict(list)
  train_ratio_list = [0.5, 0.9]
  for train_ratio in train_ratio_list:
    for samples, labels in repeat_data:
      training_size = int(train_ratio * samples.shape[0])
      samples_train = samples[:training_size, :]
      labels_train = labels[:training_size]
      samples_test = samples[training_size:, :]
      labels_test = labels[training_size:]
      labels_test_list = [[] for _ in range(labels_test.shape[0])]
      for row in range(len(labels_test)):
        for col in range(39):
          if labels_test[row][col]:
            labels_test_list[row].append(col)

      classifier = MultiLabelClassifier(LogisticRegression())
      classifier.fit(samples_train, labels_train)
      topk_list = [len(l) for l in labels_test_list]
      preds = classifier.predict_topk(samples_test, topk_list)

      results = {}
      averages = ["micro", "macro"]
      binarizer = MultiLabelBinarizer(range(CLASS_NUM))
      for avg in averages:
        results[avg] = f1_score(binarizer.fit_transform(labels_test_list),
                                binarizer.fit_transform(preds),
                                average=avg)

      all_ratio_results[train_ratio].append(results)

  for train_ratio in sorted(all_ratio_results.keys()):
    print ('Train Ratio:', train_ratio)
    for idx, result in enumerate(all_ratio_results[train_ratio]):
      print("Shuffle {}, F1 score{}".format(idx, result))
    avg_score = defaultdict(float)
    for score_dict in all_ratio_results[train_ratio]:
      for metric, score in iteritems(score_dict):
        avg_score[metric] += score
    for metric in avg_score:
      avg_score[metric] /= len(all_ratio_results[train_ratio])
    print ('Average F1 score:', dict(avg_score))


def main():
  group_edges_path = \
    '../data/blogcatelog/BlogCatalog-dataset/data/group-edges.csv'

  parser = ArgumentParser("Calculate BlogCatalog F1 score.")
  parser.add_argument("--emb_dir", default='../tf/deepwalk',
                      help="Path of directory containing embeddings.")
  parser.add_argument("--shuffles", default=2, type=int,
                      help='Number of shuffles.')
  args = parser.parse_args()

  # load and process embeddings
  embeds = np.load(args.emb_dir + '/id_emb.npy')
  id_map = {}
  for i in range(embeds.shape[0]):
    id_map[embeds[i][0]] = i
  features_matrix = embeds[[id_map[k] for k in range(NODE_NUM)]][:, 1:]

  # load and process labels
  labels = np.zeros((NODE_NUM, CLASS_NUM))
  with open(group_edges_path) as csvfile:
    lines = csv.reader(csvfile)
    for line in lines:
      node_id = int(line[0]) - 1
      group_id = int(line[1]) - 1
      labels[node_id][group_id] = 1

  # train and test args.shuffles times.
  shuffles = []
  for _ in range(args.shuffles):
    shuffles.append(skshuffle(features_matrix, labels))
  train_and_test(shuffles)


if __name__ == "__main__":
  main()
