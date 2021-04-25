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
""" Local UT test, run with `sh test_python_ut.sh`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import unittest
import numpy as np

import graphlearn as gl
import graphlearn.python.tests.utils as utils


class NodeQueryTestCase(unittest.TestCase):
  needs_initial = True
  g = None

  def initialize(self):
    pass

  @classmethod
  def setUpClass(cls):
    gl.set_default_int_attribute(10000)
    gl.set_default_float_attribute(0.01)
    gl.set_default_string_attribute('default')

    utils.prepare_env()

  @classmethod
  def tearDownClass(cls):
    cls.g.close()

  def setUp(self):
    self.node_type_ = 'user'
    self.value_range_ = (0, 100)
    self.exist_ids_ = np.array([2, 5, 8])
    self.not_exist_ids_ = np.array([-2, -5, -8])
    self.half_exist_ids_ = np.array([2, 5, -8])
    self.default_int_attr = 10000
    self.default_float_attr = 0.01
    self.default_string_attr = 'default'

    if self.needs_initial:
      self.initialize()
    if not self.g:
      time.sleep(1)

  def tearDown(self):
    pass

  def gen_test_data(self, schema):
    return utils.gen_node_data(id_type=self.node_type_,
                               id_range=self.value_range_,
                               schema=schema)

  def check_exist_attrs(self, nodes):
    self.assertListEqual([3, 2], list(nodes.int_attrs.shape))  # [int_num, batch_size]
    self.assertListEqual([3, 1], list(nodes.float_attrs.shape))  # [float_num, batch_size]
    self.assertListEqual([3, 1], list(nodes.string_attrs.shape))  # [string_num, batch_size]

    for i, value in zip(range(3), self.exist_ids_):
      # the second int is hash value, here we just check the first one
      self.assertEqual(nodes.int_attrs[i][0], value)
      self.assertAlmostEqual(nodes.float_attrs[i][0], float(value))
      self.assertEqual(nodes.string_attrs[i][0], str(value))

  def check_not_exist_attrs(self, nodes):
    self.assertListEqual([3, 2], list(nodes.int_attrs.shape))  # [int_num, batch_size]
    self.assertListEqual([3, 1], list(nodes.float_attrs.shape))  # [float_num, batch_size]
    self.assertListEqual([3, 1], list(nodes.string_attrs.shape))  # [string_num, batch_size]

    for i in range(3):
      # the second int is hash value, here we just check the first one
      self.assertEqual(nodes.int_attrs[i][0], self.default_int_attr)
      self.assertAlmostEqual(nodes.float_attrs[i][0], self.default_float_attr)
      self.assertEqual(nodes.string_attrs[i][0], self.default_string_attr)

  def check_half_exist_attrs(self, nodes):
    self.assertListEqual([3, 2], list(nodes.int_attrs.shape))  # [int_num, batch_size]
    self.assertListEqual([3, 1], list(nodes.float_attrs.shape))  # [float_num, batch_size]
    self.assertListEqual([3, 1], list(nodes.string_attrs.shape))  # [string_num, batch_size]

    for i, value in zip(range(3), self.half_exist_ids_[:2]):
      # the second int is hash value, here we just check the first one
      self.assertEqual(nodes.int_attrs[i][0], value)
      self.assertAlmostEqual(nodes.float_attrs[i][0], float(value))
      self.assertEqual(nodes.string_attrs[i][0], str(value))

    self.assertEqual(nodes.int_attrs[-1][0], self.default_int_attr)
    self.assertAlmostEqual(nodes.float_attrs[-1][0], self.default_float_attr)
    self.assertEqual(nodes.string_attrs[-1][0], self.default_string_attr)
