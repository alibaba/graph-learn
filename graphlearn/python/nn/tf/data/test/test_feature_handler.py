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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import random
import numpy as np
import tensorflow as tf
from graphlearn.python.data.feature_spec import *
from graphlearn.python.nn.tf.data.entity import Vertex
from graphlearn.python.nn.tf.data.feature_column import *
from graphlearn.python.nn.tf.data.feature_group import *

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

    # input_shape = [f_num, batch_size]
    batch1_floats = np.array([1.0 * i for i in range(10)])
    batch2_floats = np.array([[1.0 * i, 2.0 * i] for i in range(10)])

    input1 = Vertex(floats=batch1_floats)
    input2 = Vertex(floats=batch2_floats)

    # output_shape = [batch_size, f_num]
    output1 = handler(input1)
    output2 = handler(input2)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret1 = sess.run(output1)
      ret2 = sess.run(output2)

      self.assertListEqual(list(ret1.shape), list(batch1_floats.transpose().shape))
      self.assertListEqual(list(ret1), list(batch1_floats.transpose()))
      self.assertListEqual(list(ret2.shape), list(batch2_floats.transpose().shape))
      input2_t = batch2_floats.transpose()
      for i in range(ret2.shape[0]):
        self.assertListEqual(list(ret2[i]), list(input2_t[i]))

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

    # input_shape = [f_num, batch_size]
    batch1_ints = np.array([i for i in range(10)])
    batch2_ints = np.array([[i, 2 * i] for i in range(10)])

    input1 = Vertex(ints=batch1_ints)
    input2 = Vertex(ints=batch2_ints)

    # output_shape = [batch_size, total_dim]
    output1 = handler(input1)
    output2 = handler(input2)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret1 = sess.run(output1)
      ret2 = sess.run(output2)
      self.assertListEqual(list(ret1.shape), [total_dim])  # 1d array, batch_size = 1
      self.assertListEqual(list(ret2.shape), [2, total_dim]) # 2d array, batch_size = 2

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

    # input_shape = [f_num, batch_size]
    batch1_ints = np.array([i for i in range(10)])
    batch2_ints = np.array([[i, 2 * i] for i in range(10)])

    input1 = Vertex(ints=batch1_ints)
    input2 = Vertex(ints=batch2_ints)

    # output_shape = [batch_size, total_dim]
    output1 = handler(input1)
    output2 = handler(input2)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret1 = sess.run(output1)
      ret2 = sess.run(output2)
      self.assertListEqual(list(ret1.shape), [total_dim])  # 1d array, batch_size = 1
      self.assertListEqual(list(ret2.shape), [2, total_dim]) # 2d array, batch_size = 2

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

    # input_shape = [f_num, batch_size]
    batch1_ss = np.array(
        ["just,a,test",
          "items,splited,by,comma",
          "the,third,feature"])
    batch2_ss = np.array(
        [["f1,batch1", "f1,batch2,others"],
         ["f2,batch1,others", "f2,batch2"],
         ["f3,batch1,haha", "f3,batch2,hh,kk"]])

    input1 = Vertex(strings=batch1_ss)
    input2 = Vertex(strings=batch2_ss)

    # output_shape = [batch_size, total_dim]
    output1 = handler(input1)
    output2 = handler(input2)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret1 = sess.run(output1)
      ret2 = sess.run(output2)
      self.assertListEqual(list(ret1.shape), [total_dim])  # 1d array, batch_size = 1
      self.assertListEqual(list(ret2.shape), [2, total_dim]) # 2d array, batch_size = 2

  def test_popular_cases(self):
    spec = FeatureSpec(10)
    for i in range(3):
      spec.append_dense()

    total_dim = 3
    for i in range(7):
      dim = random.randint(8, 10)
      spec.append_sparse(20 + 10 * i, dim, False)
      total_dim += dim

    handler = FeatureHandler("popular", spec)
    self.assertEqual(len(handler._float_fg), 3)
    self.assertEqual(len(handler._int_fg), 0)
    self.assertEqual(len(handler._fused_int_fg) > 0, True)
    self.assertEqual(len(handler._string_fg), 0)

    # input_shape = [f_num, batch_size]
    batch1_floats = np.array([1.0 * i for i in range(3)], dtype=np.float32)
    batch2_floats = np.array([[1.0 * i, 2.0 * i] for i in range(3)], dtype=np.float32)
    batch1_ints = np.array([i for i in range(7)])
    batch2_ints = np.array([[i, 2 * i] for i in range(7)])

    input1 = Vertex(floats=batch1_floats, ints=batch1_ints)
    input2 = Vertex(floats=batch2_floats, ints=batch2_ints)

    # output_shape = [batch_size, f_num]
    output1 = handler(input1)
    output2 = handler(input2)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret1 = sess.run(output1)
      ret2 = sess.run(output2)
      self.assertListEqual(list(ret1.shape), [total_dim])
      self.assertListEqual(list(ret2.shape), [2, total_dim])

  def test_popular_cases_of_tensor(self):
    spec = FeatureSpec(10)
    for i in range(3):
      spec.append_dense()

    total_dim = 3
    for i in range(7):
      dim = random.randint(8, 10)
      spec.append_sparse(20 + 10 * i, dim, False)
      total_dim += dim

    handler = FeatureHandler("popular_tensor", spec)
    self.assertEqual(len(handler._float_fg), 3)
    self.assertEqual(len(handler._int_fg), 0)
    self.assertEqual(len(handler._fused_int_fg) > 0, True)
    self.assertEqual(len(handler._string_fg), 0)

    # input_shape = [f_num, batch_size]
    batch1_floats = np.array([1.0 * i for i in range(3)], dtype=np.float32)
    batch2_floats = np.array([[1.0 * i, 2.0 * i] for i in range(3)], dtype=np.float32)
    batch1_ints = np.array([i for i in range(7)])
    batch2_ints = np.array([[i, 2 * i] for i in range(7)])

    input1 = Vertex(floats=tf.convert_to_tensor(batch1_floats, dtype=tf.float32),
                    ints=tf.convert_to_tensor(batch1_ints, dtype=tf.int64))
    input2 = Vertex(floats=tf.convert_to_tensor(batch2_floats, dtype=tf.float32),
                    ints=tf.convert_to_tensor(batch2_ints, dtype=tf.int64))

    # output_shape = [batch_size, f_num]
    output1 = handler(input1)
    output2 = handler(input2)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret1 = sess.run(output1)
      ret2 = sess.run(output2)
      self.assertListEqual(list(ret1.shape), [total_dim])
      self.assertListEqual(list(ret2.shape), [2, total_dim])

if __name__ == "__main__":
  unittest.main()
