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
""" Local UT test, run with `sh test_python_ut.sh`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import time

from graphlearn.python.sampler.tests.test_sampling import SamplingTestCase
import graphlearn.python.tests.utils as utils

import graphlearn as gl

class GSLSamplingTestCase(SamplingTestCase):
  def test_iterate_node_with_2hop(self):
    q = self.g.V(self._node1_type).batch(2).alias('a') \
              .outV(self._edge1_type).sample(3).by('random').alias('b') \
              .outV(self._edge2_type).sample(4).by('random').alias('c') \
              .values()
    dataset = gl.Dataset(q, 10)
    while True:
      try:
        res = dataset.next()
        utils.check_equal(list(res['a'].shape), [2])
        utils.check_equal(list(res['b'].shape), [2, 3])
        utils.check_equal(list(res['c'].shape), [2 * 3, 4])
      except gl.OutOfRangeError:
        break

  def test_iterate_edge_with_1hop(self):
    q = self.g.E(self._edge1_type).batch(4).alias("a") \
              .outV().alias("b") \
              .outV(self._edge1_type).sample(2).by("random").alias("c") \
              .values()
    dataset = gl.Dataset(q)
    while True:
      try:
        res = dataset.next()
        utils.check_equal(list(res['a'].shape), [4])
        utils.check_equal(list(res['b'].shape), [4])
        utils.check_equal(list(res['b'].int_attrs.shape), [4, 2])  # [batch_size, int_attr_num]
        utils.check_equal(list(res['c'].shape), [4, 2])
      except gl.OutOfRangeError:
        break

  def test_sample_edge(self):
    q = self.g.V(self._node1_type).batch(8).alias('a') \
              .outE(self._edge1_type).sample(3).by("random").alias('b') \
              .inV().alias('c') \
              .values()
    dataset = gl.Dataset(q)
    res = dataset.next()
    utils.check_equal(list(res['a'].shape), [8])
    utils.check_equal(list(res['b'].shape), [8, 3])
    utils.check_equal(list(res['c'].shape), [8, 3])

  def test_negative_sample(self):
    q = self.g.V(self._node1_type).batch(2).alias('a') \
              .outNeg(self._edge1_type).sample(5).by("random").alias('b') \
              .values(lambda x: (x['a'].ids, x['b'].weights))
    dataset = gl.Dataset(q)
    res = dataset.next()
    utils.check_equal(list(res[0].shape), [2])
    utils.check_equal(list(res[1].shape), [2, 5])

  def test_conditional_negative_sample(self):
    def _check_ids(pos_id, neg_ids):
      utils.check_val_equal(neg_ids[0] % 5, pos_id % 5)
      utils.check_val_equal(neg_ids[1] % 4, pos_id % 4)
      utils.check_val_equal(neg_ids[2] % 3, pos_id % 3)
      utils.check_val_equal(neg_ids[3] % 3, pos_id % 3)
    q = self.g.E(self._cond_edge_type).batch(4).alias("e") \
              .each(lambda e: (
                e.inV().alias('dst'),
                e.outV().alias('src') \
                 .outNeg(self._cond_edge_type).sample(4).by('random').where(
                   "dst",
                   condition={
                     "int_cols": [0,1], "int_props": [0.25,0.25],
                     "str_cols": [0], "str_props": [0.5]}).alias('neg'))) \
              .values()
    dataset = gl.Dataset(q)
    res = dataset.next()
    src_ids = res["src"].ids
    dst_ids = res["dst"].ids
    neg_ids = res["neg"].ids
    for idx, id in enumerate(src_ids):
      print('src_id:', id, 'dst_id:', dst_ids[idx], 'neg_ids:', neg_ids[idx])
      nbr_ids = [id+2,id+3,id+5]
      utils.check_disjoint(neg_ids[idx], nbr_ids)
      _check_ids(dst_ids[idx], neg_ids[idx])

  def test_sample_with_filter(self):
    q = self.g.E(self._edge1_type).batch(4).alias("a") \
              .each(lambda e:
                (e.inV().alias('dst'),
                 e.outV().alias('src') \
                  .outV(self._edge1_type).sample(2).by("random").filter('dst').alias("b")
                )
              ) \
              .values()
    dataset = gl.Dataset(q)
    while True:
      try:
        res = dataset.next()
        utils.check_equal(list(res['b'].shape), [4, 2])
        filter_ids = res['dst'].ids
        remained_ids = res['b'].ids
        for fid, rid in zip(filter_ids, remained_ids):
          assert fid not in rid
      except gl.OutOfRangeError:
        break

  def test_full_sample(self):
    q = self.g.V(self._node2_type).batch(4).alias('a') \
              .outV(self._edge2_type).sample(3).by("full").alias('b') \
              .values(lambda x: (x['a'].ids, x['b'].ids, x['b'].offsets))
    dataset = gl.Dataset(q)
    while True:
      try:
        src, nbrs, offsets = dataset.next()
        start = 0
        for idx, offset in enumerate(offsets):
          expected_nbrs = utils.fixed_dst_ids(src[idx], self._node1_range)
          assert offset == min(len(expected_nbrs), 3)
          utils.check_subset(nbrs[start: start + offset], expected_nbrs)
          start += offset
      except gl.OutOfRangeError:
        break


if __name__ == "__main__":
  unittest.main()
