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
import os
import numpy as np
import random
import tensorflow as tf
import time

import graphlearn as gl
import graphlearn.python.nn.tf as tfg

class DataFlowTestCase(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    pass

  @classmethod
  def tearDownClass(cls):
    pass

  def setUp(self):
    self.batch_size = 20
    self.hop0 = 5
    self.hop1 = 2
    self.neg = 3
    self.dim1 = 20
    self.dim2 = 10

    self.user_path = self.gen_user()
    self.item_path = self.gen_item()
    self.u2i_path = self.gen_edge('user', 'item')
    self.i2i_path = self.gen_edge('item', 'item')
    self.graph = self.init_graph()

  def tearDown(self):
    self.graph.close()
    os.system('rm -f {} {} {} {}'.format(self.user_path,
      self.item_path, self.u2i_path, self.i2i_path))

  def gen_user(self):
    """ user node data
    id: range(0, 100)
    attributes types are: [float, float, float, float]
    """
    def write_meta(f):
      meta = 'id:int64\tfeature:string\n'
      f.write(meta)

    def write_data(f):
      for i in range(100):
        line = '%d\t%f:%f:%f:%f\n' % (i, i * 0.1, i * 0.2, i * 0.3, i * 0.4)
        f.write(line)

    path = '%s_%d' % ('user', int(time.time() * 1000))
    with open(path, 'w') as f:
      write_meta(f)
      write_data(f)
    return path

  def gen_item(self):
    """ item node data
    id: range(0, 100)
    attributes types are: [float, string, string]
    """
    def write_meta(f):
      meta = 'id:int64\tfeature:string\n'
      f.write(meta)

    def write_data(f):
       for i in range(100):
        line = '%d\t%f:%s:%s\n' % (i, i * 0.1, "hello", str(i))
        f.write(line)

    path = '%s_%d' % ('item', int(time.time() * 1000))
    with open(path, 'w') as f:
      write_meta(f)
      write_data(f)
    return path

  def gen_edge(self, src, dst):
    """ edge data
    edge count: 300
    src: range(0, 100)
    each src has 3 out edges, whose dst are random int in range(0, 100)
    """
    def write_meta(f):
        meta = 'sid:int64\tdid:int64\n'
        f.write(meta)

    def write_data(f):
      for i in range(100):
        for _ in range(3):
          line = '%d\t%d\n' % (i, random.randint(0, 100))
          f.write(line)

    path = '%s_%s_%d' % (src, dst, int(time.time() * 1000))
    with open(path, 'w') as f:
      write_meta(f)
      write_data(f)
    return path

  def init_graph(self):
    user_attr_types = ['float'] * 4
    item_attr_types = ['float', ('string', 100), ('string', 50)]
    user_attr_dims = [None] * 4
    item_attr_dims=[None, self.dim1, self.dim2]

    g = gl.Graph() \
            .node(self.user_path, 'u', decoder=gl.Decoder(
              attr_types=user_attr_types, attr_dims=user_attr_dims)) \
            .node(self.item_path, 'i', decoder=gl.Decoder(
              attr_types=item_attr_types, attr_dims=item_attr_dims)) \
            .edge(self.u2i_path, ('u', 'i', 'u-i'), decoder=gl.Decoder()) \
            .edge(self.i2i_path, ('i', 'i', 'i-i'), decoder=gl.Decoder()) \
            .init()
    return g

  def desc_query(self):
    query = self.graph.E('u-i').batch(self.batch_size).alias('seed').each(lambda e: (
                e.inV().alias('i').outV('i-i').sample(self.hop0).by('random').alias('dst_hop1') \
                                  .outV('i-i').sample(self.hop1).by('random').alias('dst_hop2'),
                e.outV().alias('u').each(lambda v: (
                  v.outV('u-i').sample(self.hop0).by('random').alias('src_hop1') \
                   .outV('i-i').sample(self.hop1).by('random').alias('src_hop2'),
                  v.outV('u-i').sample(self.hop0).by('random').alias('extra_hop1') \
                   .outV('i-i').sample(self.hop1).by('random').alias('extra_hop2'),
                  v.outNeg('u-i').sample(self.neg).by('random').alias('neg') \
                   .outV('i-i').sample(self.hop0).by('random').alias('neg_hop1') \
                   .outV('i-i').sample(self.hop1).by('random').alias('neg_hop2'))))) \
                .values()
    return query

  def test_heterogeneous_graph(self):
    query = self.desc_query()
    df = tfg.DataFlow(query)
    u_dim = 4
    i_dim = self.dim1 + self.dim2 + 1  # 1 + 20 + 10

    # u->src_hop1->src_hop2
    ego_specific_u = df.get_ego_graph('u', neighbors=['src_hop1', 'src_hop2'])
    # u->extra_hop1->extra_hop2
    ego_extra_specific_u = df.get_ego_graph('u', neighbors=['extra_hop1', 'extra_hop2'])
    # i->dst_hop1->dst_hop2
    ego_i = df.get_ego_graph('i')
    # i->dst_hop1
    ego_specific_i = df.get_ego_graph('i', neighbors=['dst_hop1'])
    # neg->neg_hop1->neg_hop2
    ego_neg = df.get_ego_graph('neg')

    ego_specific_u_1 = ego_specific_u.forward()
    ego_extra_specific_u_1 = ego_extra_specific_u.forward()
    ego_i_1 = ego_i.forward()
    ego_specific_i_1 = ego_specific_i.forward()
    ego_neg_1 = ego_neg.forward()

    emb_feature_names = []
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      emb_feature_names.append(i.name.split(':')[0])
    prefix = 'features_i/fused_embedding_column/fused_emb_'
    if self.dim1 != self.dim2:
      self.assertListEqual(sorted(emb_feature_names),
                           sorted(['{}{}'.format(prefix, self.dim1),
                                 '{}{}'.format(prefix, self.dim2)]))
    else:
      self.assertListEqual(emb_feature_names,
                           ['{}{}'.format(prefix, self.dim1)])

    self.assertListEqual(list(ego_specific_u_1.expands), [self.hop0, self.hop1])
    self.assertListEqual(list(ego_specific_u_1.expands), [self.hop0, self.hop1])
    self.assertListEqual(list(ego_i_1.expands), [self.hop0, self.hop1])
    self.assertListEqual(list(ego_specific_i_1.expands), [self.hop0])
    self.assertListEqual(list(ego_neg_1.expands), [self.hop0, self.hop1])

    x_list = [
              # Beford forward, nodes and hops are Vertexes with raw tensors.
              ego_specific_u.nodes.ids, ego_specific_u.hop(0).ids, ego_specific_u.hop(1).ids,  # 0, 1, 2
              ego_extra_specific_u.nodes.ids, ego_extra_specific_u.hop(0).ids, ego_extra_specific_u.hop(1).ids,  # 3, 4, 5
              ego_i.nodes.ids, ego_i.hop(0).ids, ego_i.hop(1).ids,  # 6, 7, 8
              ego_specific_i.nodes.ids, ego_specific_i.hop(0).ids,  # 9, 10
              ego_neg.nodes.ids, ego_neg.hop(0).ids, ego_neg.hop(1).ids,  # 11, 12, 13

              # After forward, nodes and hops are tensors of feature columns.
              ego_specific_u_1.nodes, ego_specific_u_1.hop(0), ego_specific_u_1.hop(1),  # 14, 15, 16
              ego_extra_specific_u_1.nodes, ego_extra_specific_u_1.hop(0), ego_extra_specific_u_1.hop(1),  # 17, 18, 19
              ego_i_1.nodes, ego_i_1.hop(0), ego_i_1.hop(1),  # 20, 21, 22
              ego_specific_i_1.nodes, ego_specific_i_1.hop(0),  # 23, 24
              ego_neg_1.nodes, ego_neg_1.hop(0), ego_neg_1.hop(1) # 25, 26, 27
             ]
    uids = []
    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(tf.global_variables_initializer())
      sess.run(df.iterator.initializer)
      try:
        while True:
          ret = sess.run(x_list)
          uids.extend(list(ret[0]))
          self.assertListEqual(list(ret[0]), list(ret[3]))
          self.assertNotEqual(list(ret[2]), list(ret[5]))
          self.assertListEqual(list(ret[7]), list(ret[10]))

          self.assertListEqual(list(ret[0].shape), [self.batch_size])
          self.assertListEqual(list(ret[5].shape), [self.batch_size * self.hop0 * self.hop1])
          self.assertListEqual(list(ret[8].shape), [self.batch_size * self.hop0 * self.hop1])

          self.assertListEqual(list(ret[14].shape), [self.batch_size, u_dim])
          self.assertListEqual(list(ret[19].shape), [self.batch_size * self.hop0 * self.hop1, i_dim])

          self.assertListEqual(list(ret[20].shape), [self.batch_size, i_dim])
          self.assertListEqual(list(ret[22].shape), [self.batch_size * self.hop0 * self.hop1, i_dim])
          self.assertListEqual(list(ret[27].shape), [self.batch_size * self.neg * self.hop0 * self.hop1, i_dim])
      except tf.errors.OutOfRangeError:
        self.assertListEqual(sorted(uids), sorted(list(range(100)) * 3))  # 100 * 3 u-i edges

if __name__ == "__main__":
  unittest.main()
