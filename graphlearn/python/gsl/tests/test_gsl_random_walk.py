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

class GSLRandomWalkTestCase(SamplingTestCase):
  def test_random_walk(self):
    # gl.set_default_full_nbr_num(4)
    src = self.g.V(self._node2_type).batch(4).alias('src')
    walks = src.random_walk(self._edge3_type, 10, 1.0, 1.0).alias('walks')

    query = src.values()
    dataset = gl.Dataset(query)
    while True:
      try:
        res = dataset.next()
        print(res['src'].ids)
        print(res['walks'].ids)
      except gl.OutOfRangeError:
        break


if __name__ == "__main__":
  unittest.main()