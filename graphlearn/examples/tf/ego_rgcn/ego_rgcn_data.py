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

import argparse
import datetime
import json
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
import graphlearn.examples.tf.ego_data as ego_data

class EgoRGCNData(ego_data.EgoData):
  def __init__(self, graph, model, nbrs_num=None, sampler='random',
               train_batch_size=128, test_batch_size=128, val_batch_size=128,
               num_relations=2):
    self.num_relations = num_relations
    super().__init__(graph, model, nbrs_num, sampler,
                     train_batch_size, test_batch_size, val_batch_size)

  def query(self, graph, mask=gl.Mask.TRAIN):
    """ k-hop neighbor sampling using different relations.
      For train, the query node name are as follows:
      root: ['train']
      1-hop neighbors: ['train_hop_0_r_0', 'train_hop_0_r_1']
      2-hop neighbors: ['train_hop_0_r_0_hop_1_r_0', 'train_hop_0_r_0_hop_1_r_0', 
                        'train_hop_0_r_1_hop_1_r_0', 'train_hop_0_r_1_hop_1_r_0']
      ...
    """
    prefix = ('train', 'test', 'val')[mask.value - 1]
    if prefix == "train":
      bs = self.train_batch_size
    elif prefix == "val":
      bs = self.val_batch_size
    elif prefix == "test":
      bs = self.test_batch_size
    q = graph.V("i", mask=mask).batch(bs).alias(prefix)
    current_hop_list = [q]
    for idx, hop in enumerate(self.nbrs_num):
      next_hop_list = []
      for hop_q in current_hop_list:
        for i in range(self.num_relations):
          alias = hop_q.get_alias() + '_hop_' + str(idx) + '_r_' + str(i)
          next_hop_list.append(hop_q.outV('r_'+str(i)).sample(hop).by(self.sampler).alias(alias))
      current_hop_list = next_hop_list
    return q.values()

  def reformat_node_feature(self, data_dict, alias_list, feature_handler):
    """ Transforms and organizes the input data to a list of list,
    each element of list is also a list which consits of k-hop multi-relations
    neighbor nodes' feature tensor.
    """
    cursor = 0
    x = feature_handler.forward(data_dict[alias_list[cursor]])
    cursor += 1
    x_list = [[x]]
    
    nbr_list_len = self.num_relations
    for idx in range(len(self.nbrs_num)):
      nbr_list = []
      for i in range(nbr_list_len):
        nbr_list.append(feature_handler.forward(data_dict[alias_list[cursor]]))
        cursor += 1
      x_list.append(nbr_list)
      nbr_list_len *= self.num_relations
    return x_list
