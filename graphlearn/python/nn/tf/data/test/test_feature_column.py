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
import numpy as np
import tensorflow as tf
from graphlearn.python.nn.tf.config import conf
from graphlearn.python.nn.tf.data.feature_column import *


class FeatureColumnTestCase(unittest.TestCase):
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

  def test_embedding_column(self):
    ec1 = EmbeddingColumn("ec1", 20, 8, need_hash=False)
    input1 = tf.constant([1, 2, 3, 4, 5])
    output1 = ec1.dense(input1)

    ec2 = EmbeddingColumn("ec2", 20, 8, need_hash=True)
    input2 = tf.constant([21, 22, 23, 24, 25, 26])
    output2 = ec2.dense(input2)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret1 = sess.run(output1)
      ret2 = sess.run(output2)
      self.assertListEqual([5, 8], list(ret1.shape))
      self.assertListEqual([6, 8], list(ret2.shape))

  def test_partitioned_embedding_column(self):
    cluster_spec = {"ps": ["ps0:2222", "ps1:2222"], "worker": []}
    os.environ["CLUSTER_SPEC"] = json.dumps(cluster_spec)
    ec_part = EmbeddingColumn("ec_part", 1024, 256, need_hash=False)
    input = tf.constant([1, 2, 3, 4, 5])
    output = ec_part.dense(input)

    emb_parts = []
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      if var.name.split(':')[0].startswith("embedding_column/ec_part/"):
        emb_parts.append(var)
        self.assertEqual(conf.emb_min_slice_size, var.shape[0] * var.shape[1])
    self.assertEqual(2, len(emb_parts))

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret1 = sess.run(output)
      self.assertListEqual([5, 256], list(ret1.shape))

  @unittest.skip("Only for tfra")
  def test_dynamic_embedding_column(self):
    ec1 = DynamicEmbeddingColumn("dec1", 8, is_string=True)
    input1 = tf.constant(["1", "2", "3", "4", "5"], dtype=tf.string)
    output1 = ec1.dense(input1)

    ec2 = DynamicEmbeddingColumn("dec2", 8, is_string=False)
    input2 = tf.constant([21, 22, 23, 24, 25, 26], dtype=tf.int64)
    output2 = ec2.dense(input2)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret1 = sess.run(output1)
      ret2 = sess.run(output2)
      self.assertListEqual([5, 8], list(ret1.shape))
      self.assertListEqual([6, 8], list(ret2.shape))

  def test_numeric_column(self):
    nc1 = NumericColumn("nc1", 1.0, is_float=False)
    input1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    output1 = nc1.dense(input1)

    nc2 = NumericColumn("nc2", 2.0, is_float=True)
    input2 = tf.constant([21, 22, 23, 24, 25], dtype=tf.float32)
    output2 = nc2.dense(input2)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret1 = sess.run(output1)
      ret2 = sess.run(output2)
      self.assertListEqual([1.0, 2.0, 3.0, 4.0, 5.0], list(ret1))
      self.assertListEqual([42.0, 44.0, 46.0, 48.0, 50.0], list(ret2))

  def test_fused_embedding_column(self):
    bucket_list = [10, 20, 30]
    fec = FusedEmbeddingColumn("fec", bucket_list, 8)
    inputs = []
    # shape = [feature_num, batch_size] = [3, 4]
    inputs.append(tf.constant([1, 2, 3, 4]))
    inputs.append(tf.constant([4, 5, 6, 7]))
    inputs.append(tf.constant([7, 8, 9, 10]))
    output = fec.dense(inputs)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret = sess.run(output)
      self.assertListEqual([4, 24], list(ret.shape))

  def test_sparse_embedding_column(self):
    sec1 = SparseEmbeddingColumn("sec1", 20, 8, ',')
    input1 = tf.constant(["hi,gl,and,tf"])
    output1 = sec1.dense(input1)

    sec2 = SparseEmbeddingColumn("sec2", 50, 16, ':')
    input2 = tf.constant(["it:is:just:a:test", "the:second:in:batch"])
    output2 = sec2.dense(input2)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret1 = sess.run(output1)
      ret2 = sess.run(output2)
      self.assertListEqual([1, 8], list(ret1.shape))
      self.assertListEqual([2, 16], list(ret2.shape))

  @unittest.skip("Only for tfra")
  def test_sparse_dynamic_embedding_column(self):
    sec1 = DynamicSparseEmbeddingColumn("sdec1", 8, ',')
    input1 = tf.constant(["hi,gl,and,tf"])
    output1 = sec1.dense(input1)

    sec2 = DynamicSparseEmbeddingColumn("sdec2", 16, ':')
    input2 = tf.constant(["it:is:just:a:test", "the:second:in:batch"])
    output2 = sec2.dense(input2)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret1 = sess.run(output1)
      ret2 = sess.run(output2)
      self.assertListEqual([1, 8], list(ret1.shape))
      self.assertListEqual([2, 16], list(ret2.shape))

if __name__ == "__main__":
  unittest.main()
