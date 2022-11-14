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
from graphlearn.python.utils import parse_nbrs_num

class EgoData:
  def __init__(self, graph, model, nbrs_num=None, sampler='random',
               train_batch_size=128, test_batch_size=128, val_batch_size=128):
    self.graph = graph
    self.model = model
    self.nbrs_num = parse_nbrs_num(nbrs_num)
    self.train_batch_size = train_batch_size
    self.test_batch_size = test_batch_size
    self.val_batch_size = val_batch_size
    self.sampler = sampler

    # train
    tfg.conf.training = True
    self.query_train = self.query(self.graph, gl.Mask.TRAIN)
    self.dataset_train = tfg.Dataset(self.query_train, window=10)
    self.train_iterator = self.dataset_train.iterator
    self.train_dict = self.dataset_train.get_data_dict()
    self.train_embedding = self.model.forward(
      self.reformat_node_feature(
        self.train_dict,
        self.query_train.list_alias(),
        tfg.FeatureHandler('feature_handler', self.query_train.get_node("train").decoder.feature_spec),
      ),
      self.nbrs_num
    )

    # test
    tfg.conf.training = False
    self.query_test = self.query(self.graph, gl.Mask.TEST)
    self.dataset_test = tfg.Dataset(self.query_test, window=10)
    self.test_iterator = self.dataset_test.iterator
    self.test_dict = self.dataset_test.get_data_dict()
    self.test_embedding = self.model.forward(
      self.reformat_node_feature(
        self.test_dict,
        self.query_test.list_alias(),
        tfg.FeatureHandler('feature_handler', self.query_test.get_node("test").decoder.feature_spec),
      ),
      self.nbrs_num
    )

    # val
    tfg.conf.training = False
    self.query_val = self.query(self.graph, gl.Mask.VAL)
    self.dataset_val = tfg.Dataset(self.query_val, window=10)
    self.val_iterator = self.dataset_val.iterator
    self.val_dict = self.dataset_val.get_data_dict()
    self.val_embedding = self.model.forward(
      self.reformat_node_feature(
        self.val_dict,
        self.query_val.list_alias(),
        tfg.FeatureHandler('feature_handler', self.query_val.get_node("val").decoder.feature_spec),
      ),
      self.nbrs_num
    )

  def query(self, graph, mask=gl.Mask.TRAIN):
    """
    """

  def reformat_node_feature(self, data_dict, alias_list, feature_handler):
    """
    """
