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
import time

try:
  import torch
except ImportError:
  pass
import graphlearn as gl
import graphlearn.python.nn.pytorch as thg

class DatasetTestCase(unittest.TestCase):
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

  def test_get_th_data(self):
    query = self.desc_query()
    ds = thg.Dataset(query)
    for data in ds:
      neg = data.get('neg')
      src_hop1 = data.get('src_hop1')
      neg_ids = neg.ids
      src_hop1_ids = src_hop1.ids
      src_hop1_f_attrs = src_hop1.float_attrs
      dst_hop2_i_attrs = data.get('dst_hop2').int_attrs
      self.assertListEqual(list(neg_ids.shape), [self.batch_size * self.neg])
      self.assertListEqual(list(src_hop1_ids.shape), [self.batch_size * self.hop0])
      self.assertListEqual(list(src_hop1_f_attrs.shape), [self.batch_size * self.hop0, 1])
      self.assertListEqual(list(dst_hop2_i_attrs.shape), [self.batch_size * self.hop0 * self.hop1, 2])

    dataloader = torch.utils.data.DataLoader(ds.as_dict())
    for data in dataloader:
      neg = data['neg']
      src_hop1 = data['src_hop1']
      neg_ids = neg["ids"]
      src_hop1_ids = src_hop1["ids"]
      src_hop1_f_attrs = src_hop1["float_attrs"]
      dst_hop2_i_attrs = data['dst_hop2']["int_attrs"]
      self.assertListEqual(list(neg_ids.shape), [1, self.batch_size * self.neg])
      self.assertListEqual(list(src_hop1_ids.shape), [1, self.batch_size * self.hop0])
      self.assertListEqual(list(src_hop1_f_attrs.shape), [1, self.batch_size * self.hop0, 1])
      self.assertListEqual(list(dst_hop2_i_attrs.shape), [1, self.batch_size * self.hop0 * self.hop1, 2])

if __name__ == "__main__":
  unittest.main()
