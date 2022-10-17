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

import unittest
import random

import numpy as np
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

from graphlearn.python.data.feature_spec import *
from graphlearn.python.nn.data import Data
from graphlearn.python.nn.tf.data.feature_column import *
from graphlearn.python.nn.tf.data.feature_handler import *


class FeatureHandlerTestCase(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    pass

  @classmethod
  def tearDownClass(cls):
    pass

  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_only_floats(self):
    spec = FeatureSpec(10)
    for i in range(10):
      spec.append_dense()

    handler = FeatureHandler("only_floats", spec)
    self.assertEqual(len(handler._float_fg), 10)
    self.assertEqual(len(handler._int_fg), 0)
    self.assertEqual(len(handler._fused_int_fg), 0)
    self.assertEqual(len(handler._string_fg), 0)

    batch_floats = np.array([[1.0 * i, 2.0 * i] for i in range(10)]).transpose() # [2, 10]
    input_data = Data(floats=tf.convert_to_tensor(batch_floats, dtype=tf.float32))
    output = handler(input_data) # [2, 10]

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret = sess.run(output)
      self.assertListEqual(list(ret.shape), list(batch_floats.shape))
      for i in range(ret.shape[0]):
        self.assertListEqual(list(ret[i]), list(batch_floats[i]))

  def test_only_ints(self):
    spec = FeatureSpec(10)
    total_dim = 0
    for i in range(8):
      dim = random.randint(8, 10)
      spec.append_sparse(100 + 10 * i, dim, False)
      total_dim += dim
    spec.append_sparse(100, 4, True)  # two features need hash
    spec.append_sparse(100, 4, True)
    total_dim += 8

    handler = FeatureHandler("only_ints", spec, fuse_embedding=False)
    self.assertEqual(len(handler._float_fg), 0)
    self.assertEqual(len(handler._int_fg), 10)
    self.assertEqual(len(handler._fused_int_fg), 0)
    self.assertEqual(len(handler._string_fg), 0)

    batch_ints = np.array([[i, 2 * i] for i in range(10)]).transpose() # [2, 10]
    input_data = Data(ints=tf.convert_to_tensor(batch_ints, dtype=tf.int64))
    output = handler(input_data) # [2, total_dim]

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret = sess.run(output)
      self.assertListEqual(list(ret.shape), [2, total_dim]) # 2d array, batch_size = 2

  def test_only_ints_with_fusion(self):
    spec = FeatureSpec(10)
    total_dim = 0
    for i in range(8):
      dim = random.randint(8, 10)
      spec.append_sparse(100 + 10 * i, dim, False)
      total_dim += dim

    spec.append_sparse(100, 4, True)  # two features need hash
    spec.append_sparse(100, 4, True)
    total_dim += 8

    handler = FeatureHandler("ints_with_fusion", spec, fuse_embedding=True)
    self.assertEqual(len(handler._float_fg), 0)
    self.assertEqual(len(handler._int_fg) < 10, True)
    self.assertEqual(len(handler._fused_int_fg) > 0, True)
    self.assertEqual(len(handler._string_fg), 0)

    batch_ints = np.array([[i, 2 * i] for i in range(10)]).transpose() # [2, 10]
    input_data = Data(ints=tf.convert_to_tensor(batch_ints, dtype=tf.int64))
    output = handler(input_data) # [2, total_dim]

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret = sess.run(output)
      self.assertListEqual(list(ret.shape), [2, total_dim]) # 2d array, batch_size = 2

  def test_only_strings(self):
    spec = FeatureSpec(3)
    total_dim = 0
    for i in range(3):
      dim = random.randint(8, 10)
      spec.append_multival(10 + 10 * i, dim, ",")
      total_dim += dim

    handler = FeatureHandler("only_strings", spec)
    self.assertEqual(len(handler._float_fg), 0)
    self.assertEqual(len(handler._int_fg), 0)
    self.assertEqual(len(handler._fused_int_fg), 0)
    self.assertEqual(len(handler._string_fg), 3)

    batch_ss = np.array(
        [["f1,batch1", "f1,batch2,others"],
         ["f2,batch1,others", "f2,batch2"],
         ["f3,batch1,haha", "f3,batch2,hh,kk"]]).transpose() # [2, 3]
    input_data = Data(strings=tf.convert_to_tensor(batch_ss, dtype=tf.string))
    output = handler(input_data) # [2, total_dim]

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret = sess.run(output)
      self.assertListEqual(list(ret.shape), [2, total_dim]) # 2d array, batch_size = 2

  def test_floats_and_fused_ints(self):
    spec = FeatureSpec(10)
    for i in range(3):
      spec.append_dense()

    total_dim = 3
    for i in range(7):
      dim = random.randint(8, 10)
      spec.append_sparse(20 + 10 * i, dim, False)
      total_dim += dim

    handler = FeatureHandler("floats_and_fused_ints", spec)
    self.assertEqual(len(handler._float_fg), 3)
    self.assertEqual(len(handler._int_fg), 0)
    self.assertEqual(len(handler._fused_int_fg) > 0, True)
    self.assertEqual(len(handler._string_fg), 0)

    batch_floats = np.array([[1.0 * i, 2.0 * i] for i in range(3)], dtype=np.float32).transpose() # [2, 3]
    batch_ints = np.array([[i, 2 * i] for i in range(7)]).transpose() # [2, 7]
    input_data = Data(floats=tf.convert_to_tensor(batch_floats, dtype=tf.float32),
                      ints=tf.convert_to_tensor(batch_ints, dtype=tf.int64))
    output = handler(input_data)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret = sess.run(output)
      self.assertListEqual(list(ret.shape), [2, total_dim])

if __name__ == "__main__":
  unittest.main()