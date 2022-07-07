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
"""Creates edge table, node table for train and save embeddings
from original BlogCatelog dataset.
Used by DeepWalk.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import random

from utils import download, extract

GROUP_NUM = 39
NODES_NUM = 10312


def load_data(prefix):
  edges_path = os.path.join(prefix, 'edges.csv')
  edges = []
  with open(edges_path) as csvfile:
    lines = csv.reader(csvfile)
    for line in lines:
      edges.append((int(line[0]) - 1, int(line[1]) - 1))

  return edges

if __name__ == "__main__":
  download('https://graphlearn.oss-cn-hangzhou.aliyuncs.com/data/github/BlogCatalog-dataset.zip',
           'blogcatelog.zip')
  extract('blogcatelog.zip', 'blogcatelog')

  edges = load_data('blogcatelog/BlogCatalog-dataset/data/')

  with open("blogcatelog/edge_table", 'w') as f:
    s = 'src_id:int64\tdst_id:int64\tweight:double\n'
    f.write(s)
    for u, v in edges:
      s = '%s\t%s\t%s\n' % (u, v, 1.0)
      f.write(s)

  node_ids = [i for i in range(NODES_NUM)]
  random.shuffle(node_ids)
  with open("blogcatelog/node_table", 'w') as f:
    s = 'id:int64\tweight:double\n'
    f.write(s)
    for idx in node_ids:
      s = '%s\t%s\n' % (idx, 0.0)
      f.write(s)
