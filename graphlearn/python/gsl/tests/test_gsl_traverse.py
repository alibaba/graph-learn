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

class GSLTraverseTestCase(SamplingTestCase):
  def test_iterate_edge_with_each(self):
    q = self.g.E(self._edge1_type).batch(7).alias('a') \
              .each(
                lambda x: (
                  x.outV().alias('b').outV(self._edge1_type).sample(2).by('random').alias('d'),
                  x.inV().alias('c').outV(self._edge2_type).sample(2).by('random').alias('e')
                )) \
              .values(
                lambda x: (x['a'].int_attrs, x['d'].weights, x['e'].ids)
              )
    dataset = gl.Dataset(q)
    while True:
      try:
        self.assertLessEqual(dataset.next()[0].size, 14)
      except gl.OutOfRangeError:
        break

  def test_iterate_edge_with_each_drop_last(self):
    q = self.g.E(self._edge1_type).batch(7).alias('a') \
              .each(
                lambda x: (
                  x.outV().alias('b').outV(self._edge1_type).sample(2).by('random').alias('d'),
                  x.inV().alias('c').outV(self._edge2_type).sample(2).by('random').alias('e')
                )) \
              .values(
                lambda x: (x['a'].int_attrs, x['d'].weights, x['e'].ids)
              )
    dataset = gl.Dataset(q, drop_last=True)
    while True:
      try:
        self.assertEqual(dataset.next()[0].size, 14)
      except gl.OutOfRangeError:
        break

if __name__ == "__main__":
  unittest.main()
