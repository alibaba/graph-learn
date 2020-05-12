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


class TopkNeighborSamplingTestCase(SamplingTestCase):
  """ Test case for sample topk the neighbors.
  """
  def test_1hop_circular_padding(self):
    """ Sample topk neighbors.
    """
    gl.set_padding_mode(gl.CIRCULAR)
    ids = self._seed_node2_ids
    nbr_s = self.g.neighbor_sampler(self._edge2_type,
                                    6,
                                    strategy="topk")
    nbrs = nbr_s.get(ids)
    edges = nbrs.layer_edges(1)
    nodes = nbrs.layer_nodes(1)

    utils.check_topk_edge_ids(edges, ids,
                              (0, 100), expand_factor=6,
                              default_dst_id=self._default_dst_id,
                              padding_mode="circular")
    utils.check_half_exist_edge_weights(
        edges, default_dst_id=self._default_dst_id)

  def test_1hop_replicate_padding(self):
    """ Sample topk neighbors.
    """
    gl.set_padding_mode(gl.REPLICATE)
    ids = self._seed_node2_ids
    nbr_s = self.g.neighbor_sampler(self._edge2_type,
                                    6,
                                    strategy="topk")
    nbrs = nbr_s.get(ids)
    edges = nbrs.layer_edges(1)
    nodes = nbrs.layer_nodes(1)

    utils.check_topk_edge_ids(edges, ids,
                              (0, 100), expand_factor=6,
                              default_dst_id=self._default_dst_id)
    utils.check_half_exist_edge_weights(
        edges, default_dst_id=self._default_dst_id)


  def test_1hop_using_gremlin(self):
    """ Topk neighbor sample with gremlin-like api.
    """
    gl.set_padding_mode(gl.REPLICATE)
    ids = self._seed_node2_ids
    nbrs = self.g.V(self._node2_type, feed=ids) \
      .outE(self._edge2_type).sample(2).by("topk") \
      .inV().emit()

    edges = nbrs[1]

    utils.check_topk_edge_ids(edges, ids,
                              (0, 100), expand_factor=2,
                              default_dst_id=self._default_dst_id)
    utils.check_half_exist_edge_weights(
        edges, default_dst_id=self._default_dst_id)


if __name__ == "__main__":
  unittest.main()
