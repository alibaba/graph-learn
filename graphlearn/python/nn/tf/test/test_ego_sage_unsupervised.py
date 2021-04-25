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
import time
import numpy as np
import tensorflow as tf
import graphlearn as gl
import graphlearn.python.nn.tf as tfg
import graphlearn.python.tests.utils as utils

class EgoSAGETestCase(unittest.TestCase):
  """ Base class of sampling test.
  """
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

  def gen_user(self):
    def write_meta(f):
      meta = 'id:int64\tfeature:string\n'
      f.write(meta)

    def write_data(f):
      for i in range(120):
        line = '%d\t%f:%f:%f:%f\n' % (i, i * 0.1, i * 0.2, i * 0.3, i * 0.4)
        f.write(line)

    path = '%s/%s_%d' % ('.data_path/', 'user', int(time.time() * 1000))
    with open(path, 'w') as f:
      write_meta(f)
      write_data(f)
    return path
  
  def gen_item(self):
    def write_meta(f):
      meta = 'id:int64\tfeature:string\n'
      f.write(meta)

    def write_data(f):
      for i in range(120):
        line = '%d\t%f:%f:%d:%s\n' % (i, i * 0.1, i * 0.2, i, str(i))
        f.write(line)

    path = '%s/%s_%d' % ('.data_path/', 'item', int(time.time() * 1000))
    with open(path, 'w') as f:
      write_meta(f)
      write_data(f)
    return path

  def test_heter_sage_unsupervised(self):
    user_path = self.gen_user()
    item_path = self.gen_item()
    u2i_path = utils.gen_edge_data('user', 'item', (0, 100), (0, 100), schema=[])
    i2i_path = utils.gen_edge_data('item', 'item', (0, 100), (0, 100), schema=[])

    user_attr_types = ['float'] * 4
    item_attr_types = ['float', 'float', ('string', 10), ('string', 10)]
    user_attr_dims = [None] * 4
    item_attr_dims = [None, None, 10, 10]

    g = gl.Graph() \
          .node(user_path, 'u', decoder=gl.Decoder(attr_types=user_attr_types, attr_dims=user_attr_dims)) \
          .node(item_path, 'i', decoder=gl.Decoder(attr_types=item_attr_types, attr_dims=item_attr_dims)) \
          .edge(u2i_path, ('u', 'i', 'u-i'), decoder=gl.Decoder()) \
          .edge(i2i_path, ('i', 'i', 'i-i'), decoder=gl.Decoder()) \
          .init()

    query = g.E('u-i').batch(10).alias('seed').each(lambda e: (
      e.inV().alias('i').outV('i-i').sample(15).by('topk').alias('dst_hop1').outV('i-i').sample(10).by('topk').alias('dst_hop2'),
      e.outV().alias('u').each(lambda v: (
        v.outV('u-i').sample(15).by('edge_weight').alias('src_hop1').outV('i-i').sample(10).by('topk').alias('src_hop2'),
        v.outNeg('u-i').sample(5).by('in_degree').alias('neg').outV('i-i').sample(15).by('topk').alias('neg_hop1')\
          .outV('i-i').sample(10).by('topk').alias('neg_hop2'))))) \
      .values()
    df = tfg.DataFlow(query)

    src_dim = 4
    dst_dim = 22
    layer_ui = tfg.EgoSAGELayer("heter_ui",
                                input_dim=(src_dim, dst_dim),
                                output_dim=12,
                                agg_type="mean",
                                com_type="concat")
    layer_ii = tfg.EgoSAGELayer("heter_ii",
                                input_dim=dst_dim,
                                output_dim=12,
                                agg_type="mean",
                                com_type="concat")
    layer_uii = tfg.EgoSAGELayer("heter_uii",
                                 input_dim=(12, 12),
                                 output_dim=8,
                                 agg_type="sum",
                                 com_type="concat")
    layer_iii = tfg.EgoSAGELayer("heter_iii",
                                 input_dim=(12, 12),
                                 output_dim=8,
                                 agg_type="sum",
                                 com_type="concat")
    layer_group_1 = tfg.EgoSAGELayerGroup([layer_ui, layer_ii])
    layer_group_2 = tfg.EgoSAGELayerGroup([layer_uii])
    src_model = tfg.EgoGraphSAGE(
        [layer_group_1, layer_group_2],
        bn_fn=None,
        active_fn=tf.nn.relu,
        droput=0.1)

    layer_group_3 = tfg.EgoSAGELayerGroup([layer_ii, layer_ii])
    layer_group_4 = tfg.EgoSAGELayerGroup([layer_iii])
    dst_model = tfg.EgoGraphSAGE(
        [layer_group_3, layer_group_4],
        bn_fn=None,
        active_fn=tf.nn.relu,
        droput=0.1)

    eg_u = df.get_ego_graph('u')
    eg_i = df.get_ego_graph('i')
    eg_neg = df.get_ego_graph('neg')

    src_embeddings = src_model.forward(eg_u)
    dst_embeddings = dst_model.forward(eg_i)
    neg_embeddings = dst_model.forward(eg_neg)
    neg_embeddings = tf.reshape(neg_embeddings, [-1, 5, 8])

    lp = tfg.UnsupervisedLinkPredictor(name="unlp", dims=[8, 4])
    loss = lp.forward(src_embeddings, dst_embeddings, neg_embeddings)

    u_out_degrees = eg_u.nodes.out_degrees
    u_hop_out_degrees = eg_u.hop(0).out_degrees

    trainer = tfg.Trainer()
    trainer.minimize(loss)

    def trace(ret):
      self.assertEqual(len(ret), 5)
      self.assertEqual(list(ret[0].shape), [10, 8])
      self.assertEqual(list(ret[1].shape), [10, 8])
      self.assertEqual(list(ret[3].shape), [10])
      self.assertEqual(list(ret[4].shape), [10 * 15])

    trainer.step_to_epochs(
      df, 2,
      [src_embeddings, dst_embeddings, loss, u_out_degrees, u_hop_out_degrees],
      trace)
    trainer.close()
    g.close()


if __name__ == "__main__":
  unittest.main()