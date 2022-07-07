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
"""Preprocess PPI Dataset.
Used by unsupervised GraphSage training.
Reference:
  GraphSage Author's code https://github.com/williamleif/GraphSAGE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import numpy as np

from networkx.readwrite import json_graph
from sklearn.preprocessing import StandardScaler
from utils import download, extract


def load_data(prefix):
  G_data = json.load(open(prefix + "-G.json"))
  G = json_graph.node_link_graph(G_data)
  features = np.load(prefix + "-feats.npy")
  id_map = json.load(open(prefix + "-id_map.json"))
  id_map = {int(k): int(v) for k, v in id_map.items()}

  # feature normalization:
  train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
  train_features = features[train_ids]
  scaler = StandardScaler()
  scaler.fit(train_features)
  features = scaler.transform(features)

  return G, features


if __name__ == "__main__":
  download('https://graphlearn.oss-cn-hangzhou.aliyuncs.com/data/github/ppi.zip', 'ppi.zip')
  extract('ppi.zip', '')
  G, feats = load_data('ppi/ppi')

  # generate edge table
  with open("ppi/edge_table", 'w') as f:
    s = 'src_id:int64\tdst_id:int64\tweight:double\n'
    f.write(s)
    for u, v in G.edges():
      s = '%s\t%s\t%s\n' % (u, v, 1.0)
      f.write(s)

  # generate node table
  with open("ppi/node_table", 'w') as f:
    s = 'id:int64\tfeature:string\n'
    f.write(s)
    for idx in range(len(feats)):
      features = ":".join([str(i) for i in feats[idx].tolist()])
      s = '%s\t%s\n' % (idx, features)
      f.write(s)
