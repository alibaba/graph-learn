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
"""local training script for unsupervised GraphSage"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import json
import sys
import numpy as np
import graphlearn as gl
import tensorflow as tf

from graph_sage import GraphSage


def train(config, graph):
  print('start training....')
  def model_fn():
    return GraphSage(graph,
                     config['class_num'],
                     config['features_num'],
                     config['batch_size'],
                     categorical_attrs_desc=config['categorical_attrs_desc'],
                     hidden_dim=config['hidden_dim'],
                     in_drop_rate=config['in_drop_rate'],
                     neighs_num=config['neighs_num'],
                     full_graph_mode=config['full_graph_mode'],
                     unsupervised=config['unsupervised'],
                     neg_num=config['neg_num'],
                     agg_type=config['agg_type'],
                     node_type=config['node_type'],
                     edge_type=config['edge_type'],
                     train_node_type=config['node_type'])

  cluster = tf.train.ClusterSpec({'ps': config['ps_hosts'], 'worker': config['worker_hosts']})
  trainer = gl.DistTFTrainer(model_fn,
                             cluster_spec=cluster,
                             task_name=config['job_name'],
                             task_index=config['task_index'],
                             epoch=config['epoch'],
                             optimizer=gl.get_tf_optimizer(
                                 config['learning_algo'],
                                 config['learning_rate'],
                                 config['weight_decay']))
  if config['job_name'] == 'worker': # also graph-learn client in this example.
    trainer.train_and_evaluate()
  else:
    trainer.join()


def main():
    config = {'dataset_folder': '../../data/cora/',
            'class_num': 32,
            'features_num': 2,
            'batch_size': 10, # 10
            'categorical_attrs_desc': '',
            'hidden_dim': 256,
            'in_drop_rate': 0.5,
            'hops_num': 2,
            'neighs_num': [10, 20],
            'full_graph_mode': False,
            'agg_type': 'gcn',  # mean, sum
            'learning_algo': 'adam',
            'learning_rate': 0.005,
            'weight_decay': 0.0005,
            'epoch': 1,
            'unsupervised': True,
            'use_neg': True,
            'neg_num': 10,
            'emb_save_dir': './id_emb'}

    handle = sys.argv[1]
    task_index = sys.argv[2]
    s = base64.b64decode(handle).decode('utf-8')
    obj = json.loads(s)
    node_type = obj['node_schema'][0].split(':')[0]
    edge_type = obj['edge_schema'][0].split(':')[1]
    config['node_type'] = node_type
    config['edge_type'] = edge_type
    config['train_node_type'] = node_type
    config['task_count'] = obj['client_count']
    config['task_index'] = task_index

    # use the first half as worker, others as server
    servers = obj['server'].split(',')
    mid = config['task_count'] / 2
    config['ps_hosts'] = servers[0:mid]
    config['worker_hosts'] = servers[mid:]
    if config['task_index'] < mid:
        config['job_name'] = 'worker'
    else:
        config['job_name'] = 'server'

    if config['job_name'] == 'server':
        g = gl.init_graph_from_handle(handle, task_index)
    else:
        g = gl.get_graph_from_handle(handle, worker_index=task_index, worker_count=config['task_count'])

    gl.set_tracker_mode(0)

    train(config, g)
    test(config, g)


if __name__ == "__main__":
    main()
