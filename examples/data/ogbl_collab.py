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
"""ogbl_collab dataset.
"""
import os
import numpy as np
from ogb.linkproppred import LinkPropPredDataset

# load data
dataset = LinkPropPredDataset(name='ogbl-collab')
split_edge = dataset.get_edge_split()
train_edge, valid_edge, test_edge = split_edge['train'], split_edge['valid'], split_edge['test']
# train_edge['edge'], (1179052, 2)
# train_edge['weight'], (1179052,)
# train_edge['year'], (1179052,)
# valid_edge, 60084
# test_edge, 46329

graph = dataset[0]
num_nodes = graph['num_nodes'] # 235868
node_feat = graph['node_feat'] # shape(235868, 128)

# dump to disk
root = 'ogbl_collab/'
if not os.path.exists(root):
  os.mkdir(root)
train_table = root + 'ogbl_collab_train_edge'
val_table = root + 'ogbl_collab_val_edge'
test_table = root + 'ogbl_collab_test_edge'
node_table = root + 'ogbl_collab_node'
val_neg_table = root + 'ogbl_collab_val_edge_neg'
test_neg_table = root + 'ogbl_collab_test_edge_neg'

with open(train_table, 'w') as f:
  f.write('src_id:int64' + '\t' + 'dst_id:int64' + '\t' + 'weight:double\n')
  for i in range(len(train_edge['edge'])):
    f.write(str(train_edge['edge'][i, 0]) + '\t' + str(train_edge['edge'][i, 1])
        + '\t' + str(train_edge['weight'][i]) + '\n')

with open(val_table, 'w') as f:
  f.write('src_id:int64' + '\t' + 'dst_id:int64' + '\t' + 'weight:double\n')
  for i in range(len(valid_edge['edge'])):
    f.write(str(valid_edge['edge'][i, 0]) + '\t' + str(valid_edge['edge'][i, 1])
        + '\t' + str(valid_edge['weight'][i]) + '\n')

with open(test_table, 'w') as f:
  f.write('src_id:int64' + '\t' + 'dst_id:int64' + '\t' + 'weight:double\n')
  for i in range(len(test_edge['edge'])):
    f.write(str(test_edge['edge'][i, 0]) + '\t' + str(test_edge['edge'][i, 1])
        + '\t' + str(test_edge['weight'][i]) + '\n')

with open(node_table, 'w') as f:
  f.write('id:int64' + '\t' + 'feature:string\n')
  for i in range(num_nodes):
    f.write(str(i) + '\t' + str(':'.join(map(str, node_feat[i]))) + '\n')

with open(val_neg_table, 'w') as f:
  f.write('src_id:int64' + '\t' + 'dst_id:int64' + '\t' + 'weight:double\n')
  for i in range(len(valid_edge['edge_neg'])):
    f.write(str(valid_edge['edge_neg'][i, 0]) + '\t' + str(valid_edge['edge_neg'][i, 1])
        + '\t' + '1.0\n')

with open(test_neg_table, 'w') as f:
  f.write('src_id:int64' + '\t' + 'dst_id:int64' + '\t' + 'weight:double\n')
  for i in range(len(test_edge['edge_neg'])):
    f.write(str(test_edge['edge_neg'][i, 0]) + '\t' + str(test_edge['edge_neg'][i, 1])
        + '\t' + '1.0\n')
