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
from graphlearn.python.nn.tf.model.ego_sage import EgoGraphSAGE
from graphlearn.python.nn.tf.model.ego_sage import HomoEgoGraphSAGE
from graphlearn.python.nn.tf.app.link_predictor import *
from graphlearn.python.nn.tf.data.entity import Vertex
from graphlearn.python.nn.tf.data.ego_graph import EgoGraph
from graphlearn.python.nn.tf.layers.ego_sage_layer import EgoSAGELayer
from graphlearn.python.nn.tf.layers.ego_sage_layer import EgoSAGELayerGroup


class LinkPredictorTestCase(unittest.TestCase):
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

  def get_graph(self, hops, neg=None):
    spec = FeatureSpec(10)
    for i in range(3):
      spec.append_dense()

    total_dim = 3
    for i in range(7):
      dim = i + 1
      spec.append_sparse(20 + 10 * i, dim, False)
      total_dim += dim

    neg = 1 if not neg else int(neg)
    hops[0] = int(hops[0] * neg)
    # the centric vertices share the same spec with 2-hop neighbors
    schema = [("nodes", spec), ("nodes", spec), ("nodes", spec)]

    # batch_size = 2
    # [f_num, batch_size] = [3, 2 * neg]
    batch_floats = np.array([[1.0 * i, 2.0 * i] * neg for i in range(3)])
    batch_floats = tf.convert_to_tensor(batch_floats, dtype=tf.float32)
    # [i_num, batch_size] = [7, 2 * neg]
    batch_ints = np.array([[i, 2 * i] * neg for i in range(7)])
    batch_ints = tf.convert_to_tensor(batch_ints, dtype=tf.int64)
    # [batch_size] = [2]
    batch_labels = np.array([1, 0])
    batch_labels = tf.convert_to_tensor(batch_labels, dtype=tf.int32)
    vertices = Vertex(floats=batch_floats, ints=batch_ints, labels=batch_labels)

    # [f_num, batch_size] = [3, 2 * neg * hop0]
    hop1_floats = np.array([[1.0 * i, 2.0 * i] * hops[0] for i in range(3)])
    hop1_floats = tf.convert_to_tensor(hop1_floats, dtype=tf.float32)
    # [i_num, batch_size] = [7, 2 * neg * hop0]
    hop1_ints = np.array([[i, 2 * i] * hops[0] for i in range(7)])
    hop1_ints = tf.convert_to_tensor(hop1_ints, dtype=tf.int64)
    neighbor_hop_1 = Vertex(floats=hop1_floats, ints=hop1_ints)

    # [f_num, batch_size] = [3, 2 * neg * hop0 * hop1]
    hop2_floats = np.array([[1.0 * i, 2.0 * i] * hops[0] * hops[1] for i in range(3)])
    hop2_floats = tf.convert_to_tensor(hop2_floats, dtype=tf.float32)
    # [i_num, batch_size] = [7, 2 * neg * hop0 * hop1]
    hop2_ints = np.array([[i, 2 * i] * hops[0] * hops[1] for i in range(7)])
    hop2_ints = tf.convert_to_tensor(hop2_ints, dtype=tf.int64)
    neighbor_hop_2 = Vertex(floats=hop2_floats, ints=hop2_ints)

    hops[0] = int(hops[0] / neg)
    g = EgoGraph(vertices, [neighbor_hop_1, neighbor_hop_2], schema, hops)
    return g, total_dim

  def test_unsupervised(self):
    src_hops = [4, 5]
    dst_hops = [2, 6]
    neg = 2
    src_g, src_dim = self.get_graph(src_hops)
    dst_g, dst_dim = self.get_graph(dst_hops)
    neg_g, neg_dim = self.get_graph(dst_hops, neg)

    layer_ui = EgoSAGELayer("heter_ui",
                            input_dim=(src_dim, dst_dim),
                            output_dim=12,
                            agg_type="mean",
                            com_type="concat")
    layer_ii = EgoSAGELayer("heter_ii",
                            input_dim=dst_dim,
                            output_dim=12,
                            agg_type="mean",
                            com_type="concat")
    layer_uii = EgoSAGELayer("heter_uii",
                             input_dim=(12, 12),
                             output_dim=8,
                             agg_type="sum",
                             com_type="concat")
    layer_iii = EgoSAGELayer("heter_iii",
                             input_dim=(12, 12),
                             output_dim=8,
                             agg_type="sum",
                             com_type="concat")
    layer_group_1 = EgoSAGELayerGroup([layer_ui, layer_ii])
    layer_group_2 = EgoSAGELayerGroup([layer_uii])
    src_model = EgoGraphSAGE(
        [layer_group_1, layer_group_2],
        bn_fn=None,
        active_fn=tf.nn.relu,
        droput=0.1)

    layer_group_3 = EgoSAGELayerGroup([layer_ii, layer_ii])
    layer_group_4 = EgoSAGELayerGroup([layer_iii])
    dst_model = EgoGraphSAGE(
        [layer_group_3, layer_group_4],
        bn_fn=None,
        active_fn=tf.nn.relu,
        droput=0.1)

    src_embeddings = src_model.forward(src_g)
    dst_embeddings = dst_model.forward(dst_g)
    neg_embeddings = dst_model.forward(neg_g)
    neg_embeddings = tf.reshape(neg_embeddings, [-1, neg, 8])

    lp = UnsupervisedLinkPredictor(name="unlp", dims=[8, 4])
    loss = lp.forward(src_embeddings, dst_embeddings, neg_embeddings)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      loss = sess.run(loss)
      self.assertEqual(isinstance(loss, np.float32), True)

  def test_supervised(self):
    src_hops = [4, 5]
    dst_hops = [2, 6]
    src_g, src_dim = self.get_graph(src_hops)
    dst_g, dst_dim = self.get_graph(dst_hops)

    layer_ui = EgoSAGELayer("heter_ui_super",
                            input_dim=(src_dim, dst_dim),
                            output_dim=12,
                            agg_type="mean",
                            com_type="concat")
    layer_ii = EgoSAGELayer("heter_ii_super",
                            input_dim=dst_dim,
                            output_dim=12,
                            agg_type="mean",
                            com_type="concat")
    layer_uii = EgoSAGELayer("heter_uii_super",
                             input_dim=(12, 12),
                             output_dim=8,
                             agg_type="sum",
                             com_type="concat")
    layer_iii = EgoSAGELayer("heter_iii_super",
                             input_dim=(12, 12),
                             output_dim=8,
                             agg_type="sum",
                             com_type="concat")
    layer_group_1 = EgoSAGELayerGroup([layer_ui, layer_ii])
    layer_group_2 = EgoSAGELayerGroup([layer_uii])
    src_model = EgoGraphSAGE(
        [layer_group_1, layer_group_2],
        bn_fn=None,
        active_fn=tf.nn.relu,
        droput=0.1)

    layer_group_3 = EgoSAGELayerGroup([layer_ii, layer_ii])
    layer_group_4 = EgoSAGELayerGroup([layer_iii])
    dst_model = EgoGraphSAGE(
        [layer_group_3, layer_group_4],
        bn_fn=None,
        active_fn=tf.nn.relu,
        droput=0.1)

    src_embeddings = src_model.forward(src_g)
    dst_embeddings = dst_model.forward(dst_g)

    super_lp = SupervisedLinkPredictor(name="superlp", dims=[8, 4])
    logits, loss = super_lp.forward(
        src_embeddings, dst_embeddings, src_g.nodes.labels)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      logits, loss = sess.run([logits, loss])
      self.assertEqual(isinstance(loss, np.float32), True)
      self.assertListEqual([2], list(logits.shape)) # [batch_size]

if __name__ == "__main__":
  unittest.main()
