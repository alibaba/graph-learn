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


class NodeWeightNegavtiveSamplingTestCase(SamplingTestCase):
  """ Test cases for negative sampler.
  """
  def test_neg(self):
    """ Sample negative neighbors with node_weight of the target nodes.
    """
    expand_factor = 6
    ids = self._seed_node2_ids
    nbr_s = self.g.negative_sampler(self._node2_type,
                                    expand_factor=expand_factor,
                                    strategy="node_weight")
    nodes = nbr_s.get(ids)

    utils.check_ids(nodes.ids, [i for i in range(100, 200) if i not in ids])
    utils.check_node_type(nodes, node_type=self._node2_type)
    utils.check_node_shape(nodes, ids.size * expand_factor)


if __name__ == "__main__":
  unittest.main()
