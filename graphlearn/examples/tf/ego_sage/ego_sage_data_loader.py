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

sys.path.append("..")
import ego_data_loader as ego_data
import graphlearn as gl
import graphlearn.python.nn.tf as tfg


class EgoSAGESupervisedDataLoader(ego_data.EgoDataLoader):
  def __init__(self, graph, mask=gl.Mask.TRAIN, sampler='random',
               batch_size=128, window=10,
               node_type='i', edge_type='e', nbrs_num=None, hops_num=5):
    self._node_type = node_type
    self._edge_type = edge_type
    self._nbrs_num = nbrs_num
    self._hops_num = hops_num
    super().__init__(graph, mask, sampler, batch_size, window)


  @property
  def src_ego(self):
    prefix = ('train', 'test', 'val')[self._mask.value - 1]
    return self.get_egograph(prefix)

  def _query(self, graph):
    assert len(self._nbrs_num) == self._hops_num
    prefix = ('train', 'test', 'val')[self._mask.value - 1]
    q = graph.V(self._node_type, mask=self._mask).batch(self._batch_size).alias(prefix)
    for idx, hop in enumerate(self._nbrs_num):
      alias = prefix + '_hop' + str(idx)
      q = q.outV(self._edge_type).sample(hop).by(self._sampler).alias(alias)
    return q.values()


class EgoSAGEUnsupervisedDataLoader(ego_data.EgoDataLoader):
  def __init__(self, graph, mask=gl.Mask.TRAIN, sampler='random', neg_sampler='random',
               batch_size=128, window=10,
               node_type='i', edge_type='e', nbrs_num=None, neg_num=5):
    self._neg_sampler = neg_sampler
    self._node_type = node_type
    self._edge_type = edge_type
    self._nbrs_num = nbrs_num
    self._neg_num = neg_num
    super().__init__(graph, mask, sampler, batch_size, window)

  @property
  def src_ego(self):
    return self.get_egograph('src')

  @property
  def dst_ego(self):
    return self.get_egograph('dst')

  @property
  def neg_dst_ego(self):
    return self.get_egograph('neg_dst')

  def _query(self, graph):
    seed = graph.E('train').batch(self._batch_size).shuffle(traverse=True)
    src = seed.outV().alias('src')
    dst = seed.inV().alias('dst')
    neg_dst = src.outNeg(self._edge_type).sample(self._neg_num).by(self._neg_sampler).alias('neg_dst')
    src_ego = self.meta_path_sample(src, 'src', self._nbrs_num, self._sampler)
    dst_ego = self.meta_path_sample(dst, 'dst', self._nbrs_num, self._sampler)
    dst_neg_ego = self.meta_path_sample(neg_dst, 'neg_dst', self._nbrs_num, self._sampler)
    return seed.values()

  def meta_path_sample(self, ego, ego_name, nbrs_num, sampler):
    """ creates the meta-math sampler of the input ego.
    config:
      ego: A query object, the input centric nodes/edges
      ego_name: A string, the name of `ego`.
      nbrs_num: A list, the number of neighbors for each hop.
      sampler: A string, the strategy of neighbor sampling.
    """
    alias_list = [ego_name + '_hop_' + str(i + 1) for i in range(len(nbrs_num))]
    for nbr_count, alias in zip(nbrs_num, alias_list):
      ego = ego.outV(self._edge_type).sample(nbr_count).by(sampler).alias(alias)
    return ego
