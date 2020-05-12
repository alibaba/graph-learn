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


class RandomNeighborSamplingTestCase(SamplingTestCase):
  """ Test cases for sampling with ranom choise.
  """
  def test_1hop(self):
    """ Sample one hop of neighbors.
    """
    expand_factor = 6
    ids = self._seed_node1_ids
    nbr_s = self.g.neighbor_sampler(self._edge1_type,
                                    expand_factor=expand_factor,
                                    strategy="random")
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
    """ Sample neighbors for nodes which have no out neighbors,
    and get the default neighbor id.
    """
    expand_factor = 6
    ids = self._seed_node1_ids_with_nbr_missing
    nbr_s = self.g.neighbor_sampler(self._edge1_type,
                                    expand_factor=expand_factor,
                                    strategy="random")
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
    """ Using primative api.
    """
    expand_factor = [3, 2]
    ids = self._seed_node1_ids
    nbr_s = self.g.neighbor_sampler([self._edge1_type, self._edge2_type],
                                    expand_factor=expand_factor,
                                    strategy="random")
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

  def test_1hop_using_gremlin(self):
    """ Using gremlin-like api.
    """
    expand_factor = 6
    ids = self._seed_node1_ids
    nbrs = self.g.V(self._node1_type, feed=ids) \
      .outE(self._edge1_type).sample(expand_factor).by("random") \
      .inV().emit()

    edges = nbrs[1]
    nodes = nbrs[2]

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


  def test_2hop_using_gremlin(self):
    """ Using gremlin-like api.
    """
    expand_factor = [3, 2]
    ids = self._seed_node1_ids
    nbrs = self.g.V(self._node1_type, feed=ids) \
      .outE(self._edge1_type).sample(expand_factor[0]).by("random") \
      .inV() \
      .outE(self._edge2_type).sample(expand_factor[1]).by("random") \
      .inV().emit()

    edges = nbrs[1]
    nodes = nbrs[2]
    utils.check_fixed_edge_dst_ids(edges, dst_range=self._node2_range,
                                   expected_src_ids=ids)

    ids = nodes.ids.reshape(-1)
    edges = nbrs[3]
    nodes = nbrs[4]
    utils.check_fixed_edge_dst_ids(edges, dst_range=self._node1_range,
                                   expected_src_ids=ids)

  def test_2hop_using_gremlin_with_undirected_edge(self):
    """ Using gremlin-like api and sample neighbor on undirected edges
    whose source node and dst node has defferent type.
    """
    expand_factor = [3, 2]
    ids = self._seed_node1_ids
    nbrs = self.g.V(self._node1_type, feed=ids) \
      .outE(self._edge1_type).sample(expand_factor[0]).by("random") \
      .inV() \
      .inE(self._edge1_type).sample(expand_factor[1]).by("random") \
      .inV().emit()

    edges1 = nbrs[1]
    nodes1 = nbrs[2]
    edges2 = nbrs[3]
    nodes2 = nbrs[4]

    utils.check_fixed_edge_dst_ids(edges1, dst_range=self._node2_range,
                                   expected_src_ids=ids)

    utils.check_edge_type(edges1, self._node1_type,
                          self._node2_type, self._edge1_type)
    utils.check_edge_type(edges2, self._node2_type,
                          self._node1_type, self._edge1_type + "_reverse")
    utils.check_node_type(nodes1, self._node2_type)
    utils.check_node_type(nodes2, self._node1_type)
    utils.check_node_ids(nodes2, self._node1_ids)

  def test_2hop_using_gremlin_with_undirected_edge_homo(self):
    """ Using gremlin-like api and sample neighbor on undirected edges
    whose source node and dst node has same type.
    """
    expand_factor = [3, 2]
    ids = self._seed_node2_ids

    def repeat_fn(q, params):
      return q.outE(params[0]).sample(params[1]).by("random").inV()

    nbrs = self.g.V(self._node2_type, feed=ids) \
      .repeat(repeat_fn, 2,
              params_list=[(self._edge3_type, 3), (self._edge3_type, 2)]) \
      .emit()

    edges1 = nbrs[1]
    nodes1 = nbrs[2]
    edges2 = nbrs[3]
    nodes2 = nbrs[4]

    n = expand_factor[0]*expand_factor[1]
    for i in range(0, ids.size * 3):
      for dst_id in nodes2.ids.flatten()[i : i + 2]:
        src_id = nodes1.ids.flatten()[i]
        out_id = utils.fixed_dst_ids(src_id, self._node2_range)
        in_id = utils.fixed_dst_ids(dst_id, self._node2_range)
        utils.check_ids(src_id, out_id + in_id)


if __name__ == "__main__":
  unittest.main()
