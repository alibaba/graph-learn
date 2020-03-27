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
"""Creates edge and node table for unsupervised train, and
creates test ids and degree distribution for evaluation from
Arxiv GR-QC(http://snap.stanford.edu/data/ca-GrQc.html).
Used by LINE.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import numpy as np

from networkx.readwrite import json_graph
from utils import download, extract


def load_data(prefix):
  G_data = json.load(open(prefix + "-G.json"))
  G = json_graph.node_link_graph(G_data)
  id_map = json.load(open(prefix + "-id_map.json"))
  id_map = {int(k): int(v) for k, v in id_map.items()}
  return G, id_map

if __name__ == "__main__":
  download('http://graph-learn-dataset.oss-cn-zhangjiakou.aliyuncs.com/arxiv.zip', 'arxiv.zip')
  extract('arxiv.zip', 'arxiv')
  save_data_dir = "arxiv/"

  G, id_map = load_data("arxiv/arxiv")
  degree_dict = {node: val for (node, val) in G.degree().items()}

  test_edge_list = []
  for node, val in degree_dict.items():
    if val > 1:
      for neighbor_node in list(G.neighbors(node)):
        if degree_dict[neighbor_node] > 1:
          test_edge_list.append((node, neighbor_node))
          break

  #save test edges
  test_edge_np = np.array(test_edge_list)
  print('save arxiv-links-test.npy')
  np.save(save_data_dir + "test_ids.npy", test_edge_np)

  # save train graph nodes degree
  G.remove_edges_from(test_edge_list)
  degree_list_train = [(node, val) for (node, val) in G.degree().items()]
  print('save arvix-links-train_degree.npy')
  np.save(save_data_dir + "id_degree.npy", degree_list_train)

  # save training table.
  print('save arxiv-links-train-edge-attrs')
  with open(save_data_dir + "arxiv-links-train-edge", 'w') as f:
    s = 'src_id:int64\tdst_id:int64\tweight:double\n'
    f.write(s)
    for u, v in G.edges():
      s = '%s\t%s\t%s\n' % (u, v, 1.0)
      f.write(s)

  print('save arxiv-links-train-node-attrs')
  with open(save_data_dir + "arxiv-links-train-node-attrs", 'w') as f:
    s = 'id:int64\tfeature:string\n'
    f.write(s)
    for idx in G.nodes():
      s = '%s\t%s\n' % (idx, idx)
      f.write(s)
