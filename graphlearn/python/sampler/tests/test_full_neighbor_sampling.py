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

import graphlearn as gl
from graphlearn.python.sampler.tests.test_sampling import SamplingTestCase
import graphlearn.python.tests.utils as utils


class FullNeighborSamplingTestCase(SamplingTestCase):
  """ Test case for sample all the neighbors.
  """
  def test_1hop(self):
    """ Sample full neighbors.
    """
    ids = self._seed_node1_ids
    expand_factor = 0
    nbr_s = self.g.neighbor_sampler(self._edge1_type,
                                    expand_factor,
                                    strategy="full")
    nbrs = nbr_s.get(ids)
    edges = nbrs.layer_edges(1)
    nodes = nbrs.layer_nodes(1)

    index = 0
    for node in nodes:
      utils.check_sorted_equal(
          utils.fixed_dst_ids(ids[index], self._node2_range), node.ids)
      index += 1

  def test_1hop_with_expand_factor(self):
    """ Test case for sample 1 hop neighbor with strategy of edge_weight.
    All the src_ids have neighbors.
    """
    expand_factor = 6
    ids = self._seed_node1_ids
    nbr_s = self.g.neighbor_sampler(self._edge1_type,
                                    expand_factor,
                                    strategy="full")
    nbrs = nbr_s.get(ids)
    edges = nbrs.layer_edges(1)
    nodes = nbrs.layer_nodes(1)
    utils.check_node_ids(nodes, self._node2_ids)
    utils.check_node_type(nodes, node_type=self._node2_type)


if __name__ == "__main__":
  unittest.main()
