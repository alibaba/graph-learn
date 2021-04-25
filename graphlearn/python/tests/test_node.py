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
""" Base class for node test cases.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

import graphlearn.python.tests.utils as utils


class NodeTestCase(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    pass

  @classmethod
  def tearDownClass(cls):
    pass

  def setUp(self):
    utils.prepare_env()

    self.node_type_ = 'user'
    self.value_range_ = (0, 100)
    self.ids_ = np.array([2, 5, 8])

  def tearDown(self):
    pass

  def gen_test_data(self, schema):
    return utils.gen_node_data(id_type=self.node_type_,
                               id_range=self.value_range_,
                               schema=schema)

  def check_weights(self, nodes):
    self.assertEqual(3, nodes.weights.shape[0])
    self.assertAlmostEqual(0.2, nodes.weights[0])
    self.assertAlmostEqual(0.5, nodes.weights[1])
    self.assertAlmostEqual(0.8, nodes.weights[2])

  def check_labels(self, nodes):
    self.assertEqual(3, nodes.labels.shape[0])
    self.assertEqual(2, nodes.labels[0])
    self.assertEqual(5, nodes.labels[1])
    self.assertEqual(8, nodes.labels[2])

  def check_attrs(self, nodes):
    self.assertListEqual([3, 2], list(nodes.int_attrs.shape))  # [int_num, batch_size]
    self.assertListEqual([3, 1], list(nodes.float_attrs.shape))  # [float_num, batch_size]
    self.assertListEqual([3, 1], list(nodes.string_attrs.shape))  # [string_num, batch_size]

    for i, value in zip(range(3), self.ids_):
      # the second int is hash value, here we just check the first one
      self.assertEqual(nodes.int_attrs[i][0], value)
      self.assertEqual(nodes.float_attrs[i][0], float(value))
      self.assertEqual(nodes.string_attrs[i][0], str(value))
