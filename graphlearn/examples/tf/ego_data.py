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

class EgoDataLoader:
  def __init__(self, graph, mask=gl.Mask.TRAIN, sampler='random',
               batch_size=128, window=10):
    self._graph = graph
    if isinstance(mask, gl.Mask):
      self._mask = mask
    else:
      self._mask = gl.Mask[mask.to_upper()]
    self._sampler = sampler
    self._batch_size = batch_size

    # train
    if self._mask == gl.Mask.TRAIN:
      tfg.conf.training = True
    else:
      tfg.conf.training = False
  
    self._q = self._query(self._graph)
    self._dataset = tfg.Dataset(self._q, window=window)
    self._iterator = self._dataset.iterator
    self._data_dict = self._dataset.get_data_dict()

  @property
  def iterator(self):
    return self._iterator

  @property
  def data_dict(self):
    return self._data_dict

  def data(self, key):
    return self._data_dict[key]

  def __getitem__(self, key):
    return self.data(key)

  def get_egograph(self, key):
    return self._dataset.get_egograph(key)

  @property
  def train_ego(self):
    ''' Alias for `self.get_egograph('train')`.
    '''
    return self.get_egograph('train')

  @property
  def test_ego(self):
    ''' Alias for `self.get_egograph('test')`.
    '''
    return self.get_egograph('test')

  @property
  def src_ego(self):
    ''' Alias for `self.get_egograph('src')`.
    '''
    return self.get_egograph('src')

  @property
  def dst_ego(self):
    ''' Alias for `self.get_egograph('dst')`.
    '''
    return self.get_egograph('dst')

  @property
  def train_labels(self):
    return self._data_dict['train'].labels

  @property
  def test_labels(self):
    return self._data_dict['test'].labels

  @property
  def val_labels(self):
    return self._data_dict['val'].labels

  def as_list(self):
    prefix = ('train', 'test', 'val')[self._mask.value - 1]
    return self._format(
      self._data_dict,
      self._q.list_alias(),
      tfg.FeatureHandler('feature_handler', self._q.get_node(prefix).decoder.feature_spec),
    )

  def _query(self, graph, mask=gl.Mask.TRAIN):
    """
    """

  def _format(self, data_dict, alias_list, feature_handler):
    """
    """
