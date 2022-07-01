# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
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
"""Evaluates metrics for Recommendation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def eval_metrics(gt_ids, recall_ids):
  '''Evaluates Recall, NDCG and HitRate.

  Args:
    gt_ids: A 2D array. Ground-truth ids for a batch of trigger ids where
      each row is an array of ground-truth ids for each trigger.
    recall_ids: A 2D array. Each row is an array of recall ids for each trigger.
  Returns:
    total_recall: Total recall of per-user average.
    total_ndcg: Total NDCG.
    total_hits: Total hit count.
  Note: Each return value is the sum of the input batch data,
    and the final result should be averaged by batch size.
  '''
  total_hits = 0.0
  total_recall = 0.0
  total_ndcg = 0.0
  for idx in range(len(gt_ids)):
    gt_id_list = gt_ids[idx]
    recall = 0
    dcg = 0.0
    for i, id in enumerate(recall_ids[idx]):
      if id in gt_id_list:
        recall += 1
        dcg += 1.0 / np.log2(i + 2)
    idcg = 0.0
    for i in range(recall): # calculates ideal dcg
      idcg += 1.0 / np.log2(i + 2)
    if recall > 0:
      total_ndcg += dcg / idcg
      total_hits += 1
    total_recall += recall * 1.0 / len(gt_id_list)

  return total_recall, total_ndcg, total_hits
