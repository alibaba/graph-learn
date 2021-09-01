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


class EdgeWeightNeighborSamplingTestCase(SamplingTestCase):
  def test_1hop(self):
    """ Test case for sample 1 hop neighbor with strategy of edge_weight.
    All the src_ids have neighbors.
    """
    expand_factor = 6
    ids = self._seed_node1_ids
    nbr_s = self.g.neighbor_sampler(self._edge1_type,
                                    expand_factor=expand_factor,
                                    strategy="edge_weight")
    nbrs = nbr_s.get(ids)
    edges = nbrs.layer_edges(1)
    nodes = nbrs.layer_nodes(1)

    utils.check_fixed_edge_dst_ids(edges, dst_range=self._node2_range,
                                   expected_src_ids=ids)
    utils.check_edge_type(edges,
                          src_type=self._node1_type,
                          dst_type=self._node2_type,
                          edge_type=self._edge1_type)
    utils.check_edge_shape(edges, ids.size * expand_factor)
    utils.check_edge_attrs(edges)
    utils.check_edge_labels(edges)

    utils.check_equal(nodes.ids, edges.dst_ids)
    utils.check_node_ids(nodes, self._node2_ids)
    utils.check_node_type(nodes, node_type=self._node2_type)
    utils.check_node_shape(nodes, ids.size * expand_factor)
    utils.check_node_weights(nodes)
    utils.check_node_labels(nodes)

  def test_1hop_with_neighbor_missing(self):
    """ Test case for sample 1 hop neighbor with strategy of edge_weight.
    Some of src_ids have no neighbor.
    """
    expand_factor = 6
    ids = self._seed_node1_ids_with_nbr_missing
    nbr_s = self.g.neighbor_sampler(self._edge1_type,
                                    expand_factor=expand_factor,
                                    strategy="edge_weight")
    nbrs = nbr_s.get(ids)
    edges = nbrs.layer_edges(1)
    nodes = nbrs.layer_nodes(1)

    utils.check_fixed_edge_dst_ids(edges, dst_range=self._node2_range,
                                   expected_src_ids=ids,
                                   default_dst_id=self._default_dst_id)
    utils.check_edge_type(edges,
                          src_type=self._node1_type,
                          dst_type=self._node2_type,
                          edge_type=self._edge1_type)
    utils.check_edge_shape(edges, ids.size * expand_factor)
    utils.check_not_exist_edge_attrs(
        edges, default_int_attr=self._default_int_attr,
        default_float_attr=self._default_float_attr,
        default_string_attr=self._default_string_attr,)
    utils.check_not_exist_edge_labels(edges)

    utils.check_equal(nodes.ids, edges.dst_ids)
    utils.check_node_ids(nodes, [self._default_dst_id])
    utils.check_node_type(nodes, node_type=self._node2_type)
    utils.check_node_shape(nodes, ids.size * expand_factor)
    utils.check_not_exist_node_weights(nodes)
    utils.check_not_exist_node_labels(nodes)

  def test_2hop(self):
    """ Test case for sample 2 hop neighbor with strategy of edge_weight.
    """
    expand_factor = [3, 2]
    ids = self._seed_node1_ids
    nbr_s = self.g.neighbor_sampler([self._edge1_type, self._edge2_type],
                                    expand_factor=expand_factor,
                                    strategy="edge_weight")
    nbrs = nbr_s.get(ids)
    edges = nbrs.layer_edges(1)
    nodes = nbrs.layer_nodes(1)

    utils.check_fixed_edge_dst_ids(edges, dst_range=self._node2_range,
                                   expected_src_ids=ids)
    utils.check_edge_type(edges,
                          src_type=self._node1_type,
                          dst_type=self._node2_type,
                          edge_type=self._edge1_type)
    utils.check_edge_shape(edges, ids.size * expand_factor[0])
    utils.check_edge_attrs(edges)
    utils.check_edge_labels(edges)

    utils.check_equal(nodes.ids, edges.dst_ids)
    utils.check_node_ids(nodes, self._node2_ids)
    utils.check_node_type(nodes, node_type=self._node2_type)
    utils.check_node_shape(nodes, ids.size * expand_factor[0])
    utils.check_node_weights(nodes)
    utils.check_node_labels(nodes)

    ids = nodes.ids.reshape(-1)
    edges = nbrs.layer_edges(2)
    nodes = nbrs.layer_nodes(2)

    utils.check_fixed_edge_dst_ids(edges, dst_range=self._node1_range,
                                   expected_src_ids=ids)
    utils.check_edge_type(edges,
                          src_type=self._node2_type,
                          dst_type=self._node1_type,
                          edge_type=self._edge2_type)
    utils.check_edge_shape(edges, ids.size * expand_factor[1])
    utils.check_edge_attrs(edges)
    utils.check_edge_weights(edges)

    utils.check_equal(nodes.ids, edges.dst_ids)
    utils.check_node_ids(nodes, self._node1_ids)
    utils.check_node_type(nodes, node_type=self._node1_type)
    utils.check_node_shape(nodes, ids.size * expand_factor[1])


if __name__ == "__main__":
  unittest.main()
