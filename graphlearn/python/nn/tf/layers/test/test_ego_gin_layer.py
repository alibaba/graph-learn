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
from graphlearn.python.nn.tf.layers.ego_gin_layer import EgoGINLayer
from graphlearn.python.nn.tf.layers.ego_gin_layer import EgoGINLayerGroup

class EgoGINLayerTestCase(unittest.TestCase):
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

  def test_homogeneous_graph(self):
    # MAKE LAYERS
    dims = np.array([32, 8, 1])
    layer1 = EgoGINLayer("homo_1",
                         input_dim=dims[0],
                         output_dim=dims[1],
                         eps=0.3)
    layer2 = EgoGINLayer("homo_2",
                         input_dim=dims[1],
                         output_dim=dims[2],
                         eps=0.0)
    layer_group_1 = EgoGINLayerGroup([layer1, layer1])
    layer_group_2 = EgoGINLayerGroup([layer2])

    # MAKE GRAPH DATA
    batch_size = 3
    hops = np.array([5, 2])
    # shape = [batch_size, input_dim]
    nodes = tf.convert_to_tensor(np.array(
      np.random.random([batch_size, dims[0]]),
      dtype=np.float32))
    # shape = [batch_size * hop_1, input_dim]
    hop1 = tf.convert_to_tensor(np.array(
      np.random.random([batch_size * hops[0], dims[0]]),
      dtype=np.float32))
    # shape = [batch_size * hop_1 * hop_2, input_dim]
    hop2 = tf.convert_to_tensor(np.array(
      np.random.random([batch_size * hops[0] * hops[1], dims[0]]),
      dtype=np.float32))

    # FORWARD
    inputs = [nodes, hop1, hop2]
    # h1 = [gin(nodes, hop1), gin(hop1, hop2)]
    h1 = layer_group_1.forward(inputs, hops)
    # h2 = [gin(h1[0], h1[1])]
    h2 = layer_group_2.forward(h1, hops[:-1])

    self.assertEqual(len(h1), 2)
    self.assertEqual(len(h2), 1)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret_1_0, ret_1_1, ret_2_0 = sess.run([h1[0], h1[1], h2[0]])
      self.assertListEqual([batch_size, dims[1]], list(ret_1_0.shape))
      self.assertListEqual([batch_size * hops[0], dims[1]], list(ret_1_1.shape))
      self.assertListEqual([batch_size, dims[2]], list(ret_2_0.shape))

  def test_heterogeneous_graph(self):
    # MAKE LAYERS
    # user-vedio-item
    dims_0 = np.array([24, 32, 18])
    #                  |  / |  /
    #                  | /  | /
    dims_1 = np.array([16, 12])
    #                  |  /
    #                  | /
    dims_2 = np.array([1])

    layer_uv = EgoGINLayer("heter_uv",
                           input_dim=(dims_0[0], dims_0[1]),
                           output_dim=dims_1[0],
                           eps=0.4)
    layer_vi = EgoGINLayer("heter_vi",
                           input_dim=(dims_0[1], dims_0[2]),
                           output_dim=dims_1[1],
                           eps=0.2)
    layer_uvi = EgoGINLayer("heter_uvi",
                            input_dim=(dims_1[0], dims_1[1]),
                            output_dim=dims_2[0],
                            eps=0.0)
    layer_group_1 = EgoGINLayerGroup([layer_uv, layer_vi])
    layer_group_2 = EgoGINLayerGroup([layer_uvi])

    # MAKE GRAPH DATA
    batch_size = 3
    hops = np.array([5, 2])
    # shape = [batch_size, input_dim]
    nodes = tf.convert_to_tensor(np.array(
      np.random.random([batch_size, dims_0[0]]),
      dtype=np.float32))
    # shape = [batch_size * hop_1, input_dim]
    hop1 = tf.convert_to_tensor(np.array(
      np.random.random([batch_size * hops[0], dims_0[1]]),
      dtype=np.float32))
    # shape = [batch_size * hop_1 * hop_2, input_dim]
    hop2 = tf.convert_to_tensor(np.array(
      np.random.random([batch_size * hops[0] * hops[1], dims_0[2]]),
      dtype=np.float32))

    # FORWARD
    inputs = [nodes, hop1, hop2]
    # h1 = [agg(nodes, hop1), agg(hop1, hop2)]
    h1 = layer_group_1.forward(inputs, hops)
    # h2 = [agg(h1[0], h1[1])]
    h2 = layer_group_2.forward(h1, hops[:-1])

    self.assertEqual(len(h1), 2)
    self.assertEqual(len(h2), 1)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret_1_0, ret_1_1, ret_2_0 = sess.run([h1[0], h1[1], h2[0]])
      self.assertListEqual([batch_size, dims_1[0]], list(ret_1_0.shape))
      self.assertListEqual([batch_size * hops[0], dims_1[1]], list(ret_1_1.shape))
      self.assertListEqual([batch_size, dims_2[0]], list(ret_2_0.shape))

if __name__ == "__main__":
  unittest.main()
