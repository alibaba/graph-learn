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
"""Distributed training script for supervised GraphSage.
This simple example uses two machines and each has one TensorFlow worker and ps.
Graph-learn client is colocate with TF worker, and server with ps.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import graphlearn as gl
import tensorflow as tf
from graph_sage import GraphSage
import numpy as np
import time
import os.path

# tf settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("task_index", None, "Task index")
flags.DEFINE_string("job_name", None, "worker or ps")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("tracker", '/mnt/nfs/distributed/','tracker dir')

# Note: tracker dir should be cleaned up before training.
# graphlearn settings
graph_cluster = {"client_count": 2, "tracker": FLAGS.tracker, "server_count": 2}


def load_graph(config):
  dataset_folder = config['dataset_folder']
  node_type = config['node_type']
  edge_type = config['edge_type']
  g = gl.Graph() \
    .node(dataset_folder + "node_table_degree", node_type=node_type,
          decoder=gl.Decoder(attr_types=["float"] * (config['features_num']), attr_delimiter=":")) \
    .edge(dataset_folder + "edge_table", edge_type=(node_type, node_type, edge_type),
          decoder=gl.Decoder(weighted=False), directed=False) \
    .node(dataset_folder + "node_table_degree", node_type="train",
          decoder=gl.Decoder(attr_types=["float"] * (config['features_num']), attr_delimiter=":"))
  return g

def train(config, graph):
  print ("start to train")
  def model_fn():
    return GraphSage(graph,
                     config['class_num'],
                     config['features_num'],
                     config['batch_size'],
                     val_batch_size=config['val_batch_size'],
                     test_batch_size=config['test_batch_size'],
                     categorical_attrs_desc=config['categorical_attrs_desc'],
                     hidden_dim=config['hidden_dim'],
                     in_drop_rate=config['in_drop_rate'],
                     neighs_num=config['neighs_num'],
                     agg_type=config['agg_type'],
                     full_graph_mode=config['full_graph_mode'])

  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  worker_num = len(worker_hosts)
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  trainer = gl.DistTFTrainer(model_fn,
                             cluster_spec=cluster,
                             task_name=FLAGS.job_name,
                             task_index=FLAGS.task_index,
                             epoch=config['epoch'],
                             optimizer=gl.get_tf_optimizer(
                                 config['learning_algo'],
                                 config['learning_rate'],
                                 config['weight_decay']))
  if FLAGS.job_name == 'worker': # also graph-learn client in this example.
    trainer.train()
    embs = trainer.get_node_embedding()
    np.save(config['emb_save_dir']+str(FLAGS.task_index), embs)
    fout = open("ready_status_"+str(FLAGS.task_index), "w")
    fout.close()
    print("embds shape:", embs.shape)
    ready_num = 0
    while True:
        if ready_num == worker_num:
            break
        if os.path.exists('./ready_status_'+str(ready_num)):
            ready_num = ready_num + 1
        else:
            time.sleep(1)
    print("start to test...")
    test(config, graph, worker_num)
  else:
    trainer.join()
  
  #embs = trainer.get_node_embedding()
  #print("embds shape:", embs.shape)

from sklearn.metrics.pairwise import cosine_similarity
def test(config, graph, worker_num):

  for i in range(worker_num):
      if i == 0:
          embs = np.load(config['emb_save_dir'] + str(i) + ".npy") 
      else:
          tmp = np.load(config['emb_save_dir'] + str(i) + ".npy")
          embs = np.concatenate((embs, tmp), axis = 0)
  print("concat embs shape:", embs.shape) 
  test_count = 0.0
  find_count = 0.0
  id_map = {}
  for i in range(embs.shape[0]):
      id_map[embs[i][0]] = i
  sampler_edge = graph.edge_sampler(config['edge_type'], 1000)
  edges_list = sampler_edge.get()
  src_ids = edges_list.src_ids
  dst_ids = edges_list.dst_ids
  for j in range(len(src_ids)):
      if True:
          vec_a = embs[id_map[src_ids[j]]][1:]
          vec_b = embs[id_map[dst_ids[j]]][1:]
          num = float(np.sum(vec_a * vec_b))
          denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
          if denom == 0:
              cos = 0
          else:
              cos = num / denom
          cos_sim = 0.5 + 0.5 * cos
          test_count = test_count + 1
          if cos_sim >= 0.7:
              find_count = find_count + 1
  acc = find_count/test_count
  print("Evaluation Results:")
  print("Predicted Edges in Dataset: %d / %d; Precision: %4f" %(find_count, test_count, acc))

def main():
  print("main")
  os.system('rm -f ready_status_**')
  config = {'dataset_folder': '../ldbc_10k_people/',
            'class_num': 32,
            'features_num': 100,
            'batch_size': 200, # total 140
            'val_batch_size': 10, # total 300
            'test_batch_size': 10, # total 1000
            'categorical_attrs_desc': '',
            'hidden_dim': 50,
            'in_drop_rate': 0.5,
            'hops_num': 2,
            'neighs_num': [10, 20], # [25, 10]
            'full_graph_mode': False,
            'learning_algo': 'adam',
            'learning_rate': 0.005,
            'weight_decay': 0.0005,
            'agg_type': 'gcn',
            'epoch': 3,
            'emb_save_dir': './id_emb',
            'node_type': 'item',
            'edge_type': 'relation'}

  g = load_graph(config)
  print("load_graph")
  g_role = "server"
  if FLAGS.job_name == "worker":
    g_role = "client"
  print("g_role")
  g.init(cluster=graph_cluster, job_name=g_role, task_index=FLAGS.task_index)
  train(config, g)

if __name__ == "__main__":
  main()
