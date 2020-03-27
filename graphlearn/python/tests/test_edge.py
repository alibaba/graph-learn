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
""" Base class of edge test cases.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

import graphlearn.python.tests.utils as utils


class EdgeTestCase(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    pass

  @classmethod
  def tearDownClass(cls):
    pass

  def setUp(self):
    utils.prepare_env()

    self.src_type_ = 'user'
    self.dst_type_ = 'item'
    self.edge_type_ = "first"
    self.edge_tuple_ = (self.src_type_, self.dst_type_, self.edge_type_)
    self.edge_type_list_ = ["first", "second"]
    self.src_range_ = (0, 100)
    self.dst_range_ = (100, 200)
    self.src_ids_ = np.array([2, 5, 8])
    self.dst_ids_ = np.array([102, 105, 108])
    self.batch_size_ = 3

  def tearDown(self):
    pass

  def gen_test_data(self, schema, mixed):
    path = utils.gen_edge_data(src_type=self.src_type_,
                               dst_type=self.dst_type_,
                               src_range=self.src_range_,
                               dst_range=self.dst_range_,
                               schema=schema,
                               mixed=mixed,
                               func=fixed_edges)
    return path

def fixed_edges(src_id, dst_range):  #pylint: disable=unused-argument
  return [src_id + 100]
