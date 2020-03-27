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
""" Local UT test, run with `sh test_python_ut.sh`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from graphlearn.python.sampler.tests.test_sampling import SamplingTestCase
import graphlearn.python.tests.utils as utils


class InDegreeNegavtiveSamplingTestCase(SamplingTestCase):
  """ Test cases for negative sampler.
  """
  @unittest.skip("Not always right")
  def test_neg(self):
    """ Sample negative neighbors with in-degree of the target nodes.
    """
    expand_factor = 6
    ids = self._seed_node1_ids
    nbr_s = self.g.negative_sampler(self._edge1_type,
                                    expand_factor=expand_factor,
                                    strategy="in_degree")
    nodes = nbr_s.get(ids)

    for i, e in enumerate(ids):
      expected_ids = [iid for iid in self._node2_ids if \
          iid not in utils.fixed_dst_ids(e, self._node2_range)]
      utils.check_ids(nodes.ids[i * expand_factor: (i + 1) * expand_factor],
                      expected_ids)

    utils.check_node_type(nodes, node_type=self._node2_type)
    utils.check_node_shape(nodes, ids.size * expand_factor)

  @unittest.skip("Not always right")
  def test_neg_using_gremlin(self):
    """ Using gremlin-like api.
    """
    expand_factor = 6
    ids = self._seed_node1_ids
    nbrs = self.g.V(self._node1_type, feed=ids) \
      .outNeg(self._edge1_type).sample(expand_factor).by("in_degree") \
      .emit()

    nodes = nbrs[1]

    for i, e in enumerate(ids):
      expected_ids = [iid for iid in self._node2_ids if \
          iid not in utils.fixed_dst_ids(e, self._node2_range)]
      utils.check_ids(nodes.ids[i * expand_factor: (i + 1) * expand_factor],
                      expected_ids)

    utils.check_node_type(nodes, node_type=self._node2_type)
    utils.check_node_shape(nodes, ids.size * expand_factor)


if __name__ == "__main__":
  unittest.main()
