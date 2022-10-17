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

import numpy as np
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

from graphlearn.python.nn.tf.layers.ego_rgcn_conv import EgoRGCNConv


class EgoRGCNTestCase(unittest.TestCase):
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

  def test_base_decomposition(self):
    batch_size = 32
    in_dim = [8, 16]
    out_dim = 8
    num_relations = 2
    hop1 = 5
    rgcn = EgoRGCNConv('rgcn1', in_dim, out_dim, num_relations, num_bases=2, num_blocks=None)
    x = tf.convert_to_tensor(np.array(
      np.random.random([batch_size, in_dim[0]]), dtype=np.float32))
    hop1_1 = tf.convert_to_tensor(np.array(
      np.random.random([batch_size * hop1, in_dim[1]]),
      dtype=np.float32))
    hop1_2 = tf.convert_to_tensor(np.array(
      np.random.random([batch_size * hop1, in_dim[1]]),
      dtype=np.float32))   
    neighbor = [hop1_1, hop1_2]
    out = rgcn.forward(x, neighbor, hop1)
    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      out = sess.run(out)
      self.assertListEqual([batch_size, out_dim], list(out.shape))

  def test_block_decomposition(self):
    batch_size = 32
    in_dim = [8, 16]
    out_dim = 8
    num_relations = 2
    hop1 = 5
    rgcn = EgoRGCNConv('rgcn2', in_dim, out_dim, num_relations, num_bases=None, num_blocks=2)
    x = tf.convert_to_tensor(np.array(
      np.random.random([batch_size, in_dim[0]]), dtype=np.float32))
    hop1_1 = tf.convert_to_tensor(np.array(
      np.random.random([batch_size * hop1, in_dim[1]]),
      dtype=np.float32))
    hop1_2 = tf.convert_to_tensor(np.array(
      np.random.random([batch_size * hop1, in_dim[1]]),
      dtype=np.float32))   
    neighbor = [hop1_1, hop1_2]
    out = rgcn.forward(x, neighbor, hop1)
    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      out = sess.run(out)
      self.assertListEqual([batch_size, out_dim], list(out.shape))

  def test_no_regularization(self):
    batch_size = 32
    in_dim = [8, 16]
    out_dim = 8
    num_relations = 2
    hop1 = 5
    rgcn = EgoRGCNConv('rgcn3', in_dim, out_dim, num_relations)
    x = tf.convert_to_tensor(np.array(
      np.random.random([batch_size, in_dim[0]]), dtype=np.float32))
    hop1_1 = tf.convert_to_tensor(np.array(
      np.random.random([batch_size * hop1, in_dim[1]]),
      dtype=np.float32))
    hop1_2 = tf.convert_to_tensor(np.array(
      np.random.random([batch_size * hop1, in_dim[1]]),
      dtype=np.float32))   
    neighbor = [hop1_1, hop1_2]
    out = rgcn.forward(x, neighbor, hop1)
    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      out = sess.run(out)
      self.assertListEqual([batch_size, out_dim], list(out.shape))


if __name__ == "__main__":
  unittest.main()