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


class RandomWithoutReplacementNeighborSamplingTestCase(SamplingTestCase):
  """ Test cases for sampling with ranom choise.
  """
  def test_1hop_circular_padding(self):
    """ Sample one hop of neighbors.
    """
    gl.set_padding_mode(gl.CIRCULAR)
    expand_factor = 6
    ids = self._seed_node1_ids
    nbr_s = self.g.neighbor_sampler(self._edge1_type,
                                    expand_factor=expand_factor,
                                    strategy="random_without_replacement")
    nbrs = nbr_s.get(ids)
    edges = nbrs.layer_edges(1)
    nodes = nbrs.layer_nodes(1)

    for iid, nbrs in zip(ids, nodes.ids):
      full_nbrs = utils.fixed_dst_ids(iid, (100, 200))
      utils.check_set_equal(nbrs, full_nbrs)

  def test_1hop_replicate_padding(self):
    """ Sample one hop of neighbors.
    """
    gl.set_padding_mode(gl.REPLICATE)
    expand_factor = 6
    ids = self._seed_node1_ids
    nbr_s = self.g.neighbor_sampler(self._edge1_type,
                                    expand_factor=expand_factor,
                                    strategy="random_without_replacement")
    nbrs = nbr_s.get(ids)
    edges = nbrs.layer_edges(1)
    nodes = nbrs.layer_nodes(1)

    for iid, nbrs in zip(ids, nodes.ids):
      full_nbrs = utils.fixed_dst_ids(iid, (100, 200))
      full_nbrs.extend([-1])
      utils.check_set_equal(nbrs, full_nbrs)

  def test_1hop_with_neighbor_missing(self):
    """ Sample neighbors for nodes which have no out neighbors,
    and get the default neighbor id.
    """
    gl.set_padding_mode(gl.REPLICATE)
    expand_factor = 6
    ids = self._seed_node1_ids_with_nbr_missing
    nbr_s = self.g.neighbor_sampler(self._edge1_type,
                                    expand_factor=expand_factor,
                                    strategy="random_without_replacement")
    nbrs = nbr_s.get(ids)
    edges = nbrs.layer_edges(1)
    nodes = nbrs.layer_nodes(1)

    utils.check_fixed_edge_dst_ids(edges, dst_range=self._node2_range,
                                   expected_src_ids=ids,
                                   default_dst_id=self._default_dst_id)


if __name__ == "__main__":
  unittest.main()
