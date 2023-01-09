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
from __future__ import print_function

import argparse
import datetime
import os
import sys

import numpy as np
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

import graphlearn as gl
import graphlearn.python.nn.tf as tfg

from ego_sage import EgoGraphSAGE
from ego_sage_data_loader import EgoSAGEUnsupervisedDataLoader

sys.path.append("..")
from trainer import LocalTrainer


def parse_args():
  cur_path = sys.path[0]
  argparser = argparse.ArgumentParser("Train EgoSAGE Unsupervised.")
  argparser.add_argument('--dataset_folder', type=str,
                         default=os.path.join(cur_path, '../../data/ogbl_collab/'))
  argparser.add_argument('--batch_size', type=int, default=128)
  argparser.add_argument('--features_num', type=int, default=128)
  argparser.add_argument('--hidden_dim', type=int, default=128)
  argparser.add_argument('--output_dim', type=int, default=128)
  argparser.add_argument('--nbrs_num', type=list, default=[10, 5])
  argparser.add_argument('--neg_num', type=int, default=5)
  argparser.add_argument('--learning_rate', type=float, default=0.0001)
  argparser.add_argument('--epochs', type=int, default=1)
  argparser.add_argument('--agg_type', type=str, default="mean")
  argparser.add_argument('--drop_out', type=float, default=0.0)
  argparser.add_argument('--sampler', type=str, default='random')
  argparser.add_argument('--neg_sampler', type=str, default='in_degree')
  argparser.add_argument('--temperature', type=float, default=0.07)
  argparser.add_argument('--edge_type', type=str, default='relation')
  return argparser.parse_args()

def load_graph(args):
  data_dir = args.dataset_folder
  g = gl.Graph() \
    .node(data_dir+'ogbl_collab_node', node_type='i',
          decoder=gl.Decoder(attr_types=['float'] * args.features_num,
                             attr_dims=[0]*args.features_num)) \
    .edge(data_dir+'ogbl_collab_train_edge', edge_type=('i', 'i', 'train'),
          decoder=gl.Decoder(weighted=True), directed=False)
  return g


def run(args):
  # graph input data
  g = load_graph(args=args)
  g.init()
  # Define Model
  dims = [args.features_num] + [args.hidden_dim] * (len(args.nbrs_num) - 1) + [args.output_dim]
  model = EgoGraphSAGE(dims,
                       agg_type=args.agg_type,
                       dropout=args.drop_out)

  # prepare train dataset
  train_data = EgoSAGEUnsupervisedDataLoader(g, None, args.sampler, args.neg_sampler, args.batch_size,
                                             node_type='i', edge_type='train', nbrs_num=args.nbrs_num)
  src_emb = model.forward(train_data.src_ego)
  dst_emb = model.forward(train_data.dst_ego)
  neg_dst_emb = model.forward(train_data.neg_dst_ego)
  loss = tfg.unsupervised_softmax_cross_entropy_loss(
    src_emb, dst_emb, neg_dst_emb, temperature=args.temperature)
  optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

  # train
  trainer = LocalTrainer()
  trainer.train(train_data.iterator, loss, optimizer, epochs=args.epochs)

  # finish
  g.close()

if __name__ == "__main__":
  run(parse_args())
