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
  def test_traverse_with_mask(self):
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


if __name__ == "__main__":
  unittest.main()
