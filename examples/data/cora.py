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
"""Preprocess cora dataset and generate node, edge, train, val, test table.
Used by GCN, GAT, GraphSage supervised training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.sparse as sp

from utils import download, extract

def preprocess(dataset):
  # process node table
  node_table = "{}/node_table".format(dataset)
  edge_table = "{}/edge_table".format(dataset)
  edge_table_with_self_loop = '{}/edge_table_with_self_loop'.format(dataset)
  train_table = "{}/train_table".format(dataset)
  val_table = "{}/val_table".format(dataset)
  test_table = "{}/test_table".format(dataset)

  idx_features_labels = np.genfromtxt(dataset  + "/cora.content",
                                      dtype=np.dtype(str))
  if not os.path.exists(edge_table_with_self_loop):
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

    features = sp.csr_matrix(idx_features_labels[:, 1:-1],
                             dtype=np.float32)
    features = feature_normalize(features)
    features = np.array(features.todense())
    labels = encode_label(idx_features_labels[:, -1])
    node_idxs = []

    with open(node_table, 'w') as f:
      f.write("id:int64" + "\t" + "label:int64" + "\t" + "feature:string" + "\n")
      for i in range(idx.shape[0]):
        f.write(str(idx[i]) + "\t" + str(labels[i]) +
                "\t" + str(":".join(map(str, features[i]))) + "\n")
        node_idxs.append(str(idx[i]))

    with open(train_table, 'w') as f:
      f.write("id:int64" + "\t" + "weight:float" + "\n")
      for i in range(140):
        f.write(str(idx[i]) + "\t" + str(1.0)  + "\n")
    with open(val_table, 'w') as f:
      f.write("id:int64" + "\t" + "weight:float" + "\n")
      for i in range(200, 500):
        f.write(str(idx[i]) + "\t" + str(1.0)  + "\n")
    with open(test_table, 'w') as f:
      f.write("id:int64" + "\t" + "weight:float" + "\n")
      for i in range(500, 1500):
        f.write(str(idx[i]) + "\t" + str(1.0)  + "\n")

    # process edge table
    edges = np.genfromtxt(dataset + "/cora.cites", dtype=np.int32)
    with open(edge_table, 'w') as f:
      f.write("src_id: int64" + "\t"
              + "dst_id: int64" + "\t"
              + "weight: double" + "\n")
      for i in range(edges.shape[0]):
        f.write(str(edges[i][0]) + "\t" + str(edges[i][1]) + "\t" + "0.0" + "\n")

    with open(edge_table_with_self_loop, 'w') as f:
      f.write("src_id: int64" + "\t"
              + "dst_id: int64" + "\t"
              + "weight: double" + "\n")
      for i in range(edges.shape[0]):
        if edges[i][0] != edges[i][1]:
          f.write(str(edges[i][0]) + "\t" + str(edges[i][1]) + "\t" + "0.0" + "\n")
      for idx in node_idxs:
        f.write(idx + '\t' + idx + '\t' + '0.0' + '\n')

    print("Data Process Done.")
    return
  print("Data {} has exist.".format(dataset))

def encode_label(labels):
  classes = list(sorted(set(labels)))
  classes_dict = {c: i for i, c in
                  enumerate(classes)}
  labels_int64 = np.array(list(map(classes_dict.get, labels)),
                           dtype=np.int64)
  return labels_int64

def feature_normalize(sparse_matrix):
  """Normalize sparse matrix feature by row.
  Reference:
    DGL(https://github.com/dmlc/dgl).
  """
  row_sum = np.array(sparse_matrix.sum(1))
  row_norm = np.power(row_sum, -1).flatten()
  row_norm[np.isinf(row_norm)] = 0.
  row_matrix_norm = sp.diags(row_norm)
  sparse_matrix = row_matrix_norm.dot(sparse_matrix)
  return sparse_matrix

if __name__ == "__main__":
  download('http://graph-learn-dataset.oss-cn-zhangjiakou.aliyuncs.com/cora.zip', 'cora.zip')
  extract('cora.zip', 'cora')
  preprocess('cora')
