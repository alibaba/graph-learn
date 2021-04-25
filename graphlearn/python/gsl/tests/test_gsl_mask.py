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

class GSLMaskTestCase(SamplingTestCase):
  def test_traverse_with_mask_eager_mode(self):
    gl.set_eager_mode(True)
    bs = 8
    q = self.g.V(self._node1_type, mask=gl.Mask.TRAIN).batch(bs).alias('train') \
            .values(lambda x:
               (x['train'].ids, x['train'].int_attrs, x['train'].float_attrs, x['train'].string_attrs))
    iteration = 0
    res = []
    while True:
      try:
        ids, i, f, s = q.next()
        utils.check_i_attrs(i, ids)
        utils.check_f_attrs(f, ids)
        utils.check_s_attrs(s, ids)
        iteration += 1
        res += list(ids)
      except gl.OutOfRangeError:
        break
    utils.check_sorted_equal(res, range(self._train_node_range[0], self._train_node_range[1]))

  def test_traverse_with_mask(self):
    gl.set_eager_mode(False)
    bs = 8
    q = self.g.V(self._node1_type, mask=gl.Mask.TEST).batch(bs).alias('test') \
            .values(lambda x:
               (x['test'].ids, x['test'].int_attrs, x['test'].float_attrs, x['test'].string_attrs))
    dataset = gl.Dataset(q)
    iteration = 0
    for i in range(2):
      res = []
      while True:
        try:
          ids, i, f, s = dataset.next()
          utils.check_i_attrs(i, ids)
          utils.check_f_attrs(f, ids)
          utils.check_s_attrs(s, ids)
          iteration += 1
          res += list(ids)
        except gl.OutOfRangeError:
          break
      utils.check_sorted_equal(res, range(self._test_node_range[0], self._test_node_range[1]))

  def test_sampling_with_mask_eager_mode(self):
    gl.set_eager_mode(True)
    bs = 8
    q = self.g.E(self._edge1_type, mask=gl.Mask.VAL).batch(bs).alias('val') \
              .each(
                lambda e:
                  (e.outV().alias('src'),
                   e.inV().alias('dst') \
                    .outV(self._edge2_type).sample(3).by('topk').alias('nbr'))
              ).values(lambda x:
                 (x['src'].ids,
                  x['val'].labels,
                  x['dst'].ids, x['dst'].weights, x['dst'].labels,
                  x['nbr'].ids, x['nbr'].int_attrs))
    iteration = 0
    for i in range(2):
      res = []
      while True:
        try:
          sid, elb, did, dwei, dlb, nid, ni = q.next()
          utils.check_id_weights(did, dwei)
          utils.check_equal(dlb, did)
          iteration += 1
          res += list(sid)
        except gl.OutOfRangeError:
          break
      whole = range(self._val_node_range[0], self._val_node_range[1])
      expected = []
      for elem in whole:
        expected += [elem] * len(utils.fixed_dst_ids(elem, self._node2_range))
      utils.check_sorted_equal(res, expected)

  def test_sampling_with_mask(self):
    gl.set_eager_mode(False)
    bs = 8
    q = self.g.E(self._edge1_type, mask=gl.Mask.TEST).batch(bs).alias('test') \
              .each(
                lambda e:
                  (e.outV().alias('src'),
                   e.inV().alias('dst') \
                    .outV(self._edge2_type).sample(3).by('topk').alias('nbr'))
              ).values(lambda x:
                 (x['src'].ids,
                  x['test'].labels,
                  x['dst'].ids, x['dst'].weights, x['dst'].labels,
                  x['nbr'].ids, x['nbr'].int_attrs))
    dataset = gl.Dataset(q)
    iteration = 0
    res = []
    while True:
      try:
        sid, elb, did, dwei, dlb, nid, ni = dataset.next()
        utils.check_id_weights(did, dwei)
        utils.check_equal(dlb, did)
        iteration += 1
        res += list(sid)
      except gl.OutOfRangeError:
        break
    whole = range(self._test_node_range[0], self._test_node_range[1])
    expected = []
    for elem in whole:
      expected += [elem] * len(utils.fixed_dst_ids(elem, self._node2_range))
    utils.check_sorted_equal(res, expected)


if __name__ == "__main__":
  unittest.main()
