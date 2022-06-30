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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import graphlearn as gl
import numpy as np

sys.path.append(os.path.dirname(sys.path[0]))
from data.utils import download, extract
from eval_rec_metric import eval_metrics

def load_graph(config, i_emb_table, u_emb_table=None):
  '''load embedding tables in GL that used to lookup embedding and
    do knn search.
  '''
  gl.set_knn_metric(config['knn_metric'])
  option = gl.IndexOption()
  option.name = "knn"
  option.index_type = "flat"

  if u_emb_table is None:
    g = gl.Graph()\
      .node(i_emb_table, node_type='i',
            decoder=gl.Decoder(attr_types=['float'] * config['emb_dim'], attr_delimiter=","),
            option=option)
  else:
    g = gl.Graph()\
      .node(i_emb_table, node_type='i',
            decoder=gl.Decoder(attr_types=['float'] * config['emb_dim'], attr_delimiter=","),
            option=option) \
      .node(u_emb_table, node_type='u',
            decoder=gl.Decoder(attr_types=['float'] * config['emb_dim'], attr_delimiter=","))
  return g


def main(config):
  if config['recall_type'] == 'u2i':
    g = load_graph(config, config['i_emb_table'], config['u_emb_table'])
  else:
    g = load_graph(config, config['i_emb_table'])
  g.init()
  with open(config['gt_items_table']) as f:
    gt_record = f.readlines()
    gt_record = [record.rstrip().split('\t') for record in gt_record]
    src_ids = np.array([int(src_items[0]) for src_items in gt_record])
    gt_items = [list(map(int, src_items[1].split(','))) for src_items in gt_record]
    src_type = 'u' if config['recall_type'] == 'u2i' else 'i'
    recall_ids, _ = g.search('i', g.get_nodes(src_type, src_ids).float_attrs,
                             gl.KnnOption(k=config['top_k']))
    total_recall, total_ndcg, total_hits = eval_metrics(gt_items, recall_ids)
    total_size = len(gt_items)
  print("Recall@{} {:.4f}, NDCG@{} {:.4f}, HitRate@{} {:.4f}".format(
      config['top_k'], total_recall/total_size,
      config['top_k'], total_ndcg/total_size,
      config['top_k'], total_hits/total_size))
  g.close()

if __name__ == "__main__":
  config = {'u_emb_table': 'eval_test/u_emb.txt', # required for recall_type=u2i
            'i_emb_table': 'eval_test/i_emb.txt',
            'gt_items_table': 'eval_test/gt.txt',
            'emb_dim': 128,
            'recall_type': 'u2i', # u2i or i2i
            'top_k': 20,
            'knn_metric': 1, # 1 means inner product and 0 means L2 distance.
            }
  download('https://graphlearn.oss-cn-hangzhou.aliyuncs.com/data/github/eval_test.tar.gz', 'eval_test.tar.gz')
  extract('eval_test.tar.gz', 'eval_test')
  main(config)
