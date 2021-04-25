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
from graphlearn.python.nn.tf.model.ego_gat import EgoGAT
from graphlearn.python.nn.tf.model.ego_gat import HomoEgoGAT
from graphlearn.python.nn.tf.data.entity import Vertex
from graphlearn.python.nn.tf.data.ego_graph import EgoGraph
from graphlearn.python.nn.tf.layers.ego_gat_layer import EgoGATLayer
from graphlearn.python.nn.tf.layers.ego_gat_layer import EgoGATLayerGroup

class EgoGATTestCase(unittest.TestCase):
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
    spec = FeatureSpec(10)
    for i in range(3):
      spec.append_dense()

    total_dim = 3
    for i in range(7):
      dim = random.randint(8, 10)
      spec.append_sparse(20 + 10 * i, dim, False)
      total_dim += dim

    hops = [4, 5]
    # the centric vertices share the same spec with 2-hop neighbors
    schema = [("nodes", spec), ("nodes", spec), ("nodes", spec)]

    # [f_num, batch_size] = [3, 2]
    batch_floats = np.array([[1.0 * i, 2.0 * i] for i in range(3)])
    batch_floats = tf.convert_to_tensor(batch_floats, dtype=tf.float32)
    # [i_num, batch_size] = [7, 2]
    batch_ints = np.array([[i, 2 * i] for i in range(7)])
    batch_ints = tf.convert_to_tensor(batch_ints, dtype=tf.int64)
    vertices = Vertex(floats=batch_floats, ints=batch_ints)

    # [f_num, batch_size] = [3, 2 * 4]
    hop1_floats = np.array([[1.0 * i, 2.0 * i] * hops[0] for i in range(3)])
    hop1_floats = tf.convert_to_tensor(hop1_floats, dtype=tf.float32)
    # [i_num, batch_size] = [7, 2 * 4]
    hop1_ints = np.array([[i, 2 * i] * hops[0] for i in range(7)])
    hop1_ints = tf.convert_to_tensor(hop1_ints, dtype=tf.int64)
    neighbor_hop_1 = Vertex(floats=hop1_floats, ints=hop1_ints)

    # [f_num, batch_size] = [3, 2 * 4 * 5]
    hop2_floats = np.array([[1.0 * i, 2.0 * i] * hops[0] * hops[1] for i in range(3)])
    hop2_floats = tf.convert_to_tensor(hop2_floats, dtype=tf.float32)
    # [i_num, batch_size] = [7, 2 * 4 * 5]
    hop2_ints = np.array([[i, 2 * i] * hops[0] * hops[1] for i in range(7)])
    hop2_ints = tf.convert_to_tensor(hop2_ints, dtype=tf.int64)
    neighbor_hop_2 = Vertex(floats=hop2_floats, ints=hop2_ints)

    g = EgoGraph(vertices, [neighbor_hop_1, neighbor_hop_2], schema, hops)

    dims = np.array([total_dim, 16, 8])
    model = HomoEgoGAT(
        dims,
        num_head=5,
        bn_fn=None,
        active_fn=tf.nn.relu,
        droput=0.1)
    embeddings = model.forward(g)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret = sess.run(embeddings)
      self.assertListEqual([2, 8], list(ret.shape)) # [batch_size, output_dim]

  def test_heterogeneous_graph(self):
    u_spec = FeatureSpec(10)
    for i in range(3):
      u_spec.append_dense()

    u_total_dim = 3
    for i in range(7):
      dim = random.randint(8, 10)
      u_spec.append_sparse(20 + 10 * i, dim, False)
      u_total_dim += dim

    i_spec = FeatureSpec(19)
    for i in range(6):
      i_spec.append_dense()

    i_total_dim = 6
    for i in range(13):
      dim = random.randint(8, 11)
      i_spec.append_sparse(30 + 10 * i, dim, False)
      i_total_dim += dim

    u_out_dim = 16
    i_out_dim = 12
    out_dim = 9
    hops = [4, 5]
    # the centric vertices share the same spec with 2-hop neighbors
    # metapath: u--i--i
    schema = [("u_nodes", u_spec), ("nbr", i_spec), ("nbr", i_spec)]

    # [f_num, batch_size] = [3, 2]
    batch_floats = np.array([[1.0 * i, 2.0 * i] for i in range(3)])
    batch_floats = tf.convert_to_tensor(batch_floats, dtype=tf.float32)
    # [i_num, batch_size] = [7, 2]
    batch_ints = np.array([[i, 2 * i] for i in range(7)])
    batch_ints = tf.convert_to_tensor(batch_ints, dtype=tf.int64)
    vertices = Vertex(floats=batch_floats, ints=batch_ints)

    # [f_num, batch_size] = [6, 2 * 4]
    hop1_floats = np.array([[1.0 * i, 2.0 * i] * hops[0] for i in range(6)])
    hop1_floats = tf.convert_to_tensor(hop1_floats, dtype=tf.float32)
    # [i_num, batch_size] = [13, 2 * 4]
    hop1_ints = np.array([[i, 2 * i] * hops[0] for i in range(13)])
    hop1_ints = tf.convert_to_tensor(hop1_ints, dtype=tf.int64)
    neighbor_hop_1 = Vertex(floats=hop1_floats, ints=hop1_ints)

    # [f_num, batch_size] = [6, 2 * 4 * 5]
    hop2_floats = np.array([[1.0 * i, 2.0 * i] * hops[0] * hops[1] for i in range(6)])
    hop2_floats = tf.convert_to_tensor(hop2_floats, dtype=tf.float32)
    # [i_num, batch_size] = [13, 2 * 4 * 5]
    hop2_ints = np.array([[i, 2 * i] * hops[0] * hops[1] for i in range(13)])
    hop2_ints = tf.convert_to_tensor(hop2_ints, dtype=tf.int64)
    neighbor_hop_2 = Vertex(floats=hop2_floats, ints=hop2_ints)

    g = EgoGraph(vertices, [neighbor_hop_1, neighbor_hop_2], schema, hops)

    layer_ui = EgoGATLayer("heter_uv",
                           input_dim=(u_total_dim, i_total_dim),
                           output_dim=u_out_dim,
                           num_head=5)
    layer_ii = EgoGATLayer("heter_ii",
                           input_dim=i_total_dim,
                           output_dim=i_out_dim,
                           num_head=5)
    layer_uii = EgoGATLayer("heter_uii",
                            input_dim=(u_out_dim, i_out_dim),
                            output_dim=out_dim,
                            num_head=5)
    layer_group_1 = EgoGATLayerGroup([layer_ui, layer_ii])
    layer_group_2 = EgoGATLayerGroup([layer_uii])

    model = EgoGAT(
        [layer_group_1, layer_group_2],
        bn_fn=None,
        active_fn=tf.nn.relu,
        droput=0.1)
    embeddings = model.forward(g)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      ret = sess.run(embeddings)
      self.assertListEqual([2, 9], list(ret.shape)) # [batch_size, output_dim]

if __name__ == "__main__":
  unittest.main()
