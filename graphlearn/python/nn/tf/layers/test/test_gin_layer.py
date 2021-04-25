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
from graphlearn.python.data.feature_spec import FeatureSpec
from graphlearn.python.nn.tf.data.entity import Vertex
from graphlearn.python.nn.tf.data.subgraph import SubGraph
from graphlearn.python.nn.tf.layers.gin_layer import GINLayer

class GINLayerTestCase(unittest.TestCase):
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

  @unittest.skip("Temporaray skip")
  def get_sub_graph_data(self):
    spec = FeatureSpec(10)
    for i in range(3):
      spec.append_dense()

    total_dim = 3
    for i in range(7):
      dim = random.randint(8, 10)
      spec.append_sparse(20 + 10 * i, dim, False)
      total_dim += dim

    N = 10
    # [f_num, batch_size] = [3, N]
    batch_floats = np.random.random([3, N])
    batch_floats = tf.convert_to_tensor(batch_floats, dtype=tf.float32)
    # [i_num, batch_size] = [7, N]
    batch_ints = np.array([[i * j for j in range(N)] for i in range(7)])
    batch_ints = tf.convert_to_tensor(batch_ints, dtype=tf.int64)

    vertices = Vertex(floats=batch_floats, ints=batch_ints)

    adj = np.zeros([N, N], dtype=np.float32)
    for i in range(N):
      for j in range(N):
        adj[i][j] = i

    tf_adj = tf.convert_to_tensor(adj, dtype=tf.float32)

    g = SubGraph(vertices, tf_adj, schema=("nodes", spec))
    return g, N, total_dim

  def test_homogeneous_graph(self):
    g, N, total_dim = self.get_sub_graph_data()

    dims = np.array([total_dim, 8, 2])
    layer1 = GINLayer("gin_1",
                      input_dim=dims[0],
                      output_dim=dims[1],
                      eps=0.2,
                      use_bias=True)
    layer2 = GINLayer("gin_2",
                      input_dim=dims[1],
                      output_dim=dims[2],
                      eps=0.0,
                      use_bias=False)

    g = g.forward()
    output1 = layer1(g.nodes, g)
    output2 = layer2(output1, g)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret1, ret2 = sess.run([output1, output2])

      self.assertListEqual(list(ret1.shape), [N, 8])
      self.assertListEqual(list(ret2.shape), [N, 2])

if __name__ == "__main__":
  unittest.main()
