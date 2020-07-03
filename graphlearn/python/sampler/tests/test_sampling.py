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
""" Base sampling test class.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import unittest
import numpy as np

import graphlearn as gl
import graphlearn.python.tests.utils as utils


class SamplingTestCase(unittest.TestCase):
  """ Base class of sampling test.
  """
  needs_initial = True
  g = None

  def initialize(self):
    """ Init gl.Graph
    """
    node1_path = utils.gen_node_data([self._node1_type], [self._node1_range],
                                     [utils.ATTRIBUTED])[0]
    node2_path = utils.gen_node_data([self._node2_type], [self._node2_range],
                                     [utils.WEIGHTED, utils.LABELED])[0]
    edge1_path = utils.gen_edge_data(self._node1_type, self._node2_type,
                                     self._node1_range, self._node2_range,
                                     schema=[utils.ATTRIBUTED, utils.LABELED])
    edge2_path = utils.gen_edge_data(self._node2_type, self._node1_type,
                                     self._node2_range, self._node1_range,
                                     schema=[utils.ATTRIBUTED, utils.WEIGHTED])
    edge3_path = utils.gen_edge_data(self._node2_type, self._node2_type,
                                     self._node2_range, self._node2_range,
                                     schema=[utils.WEIGHTED])

    self.__class__.needs_initial = False
    self.__class__.g = gl.Graph() \
      .node(source=node1_path, node_type=self._node1_type,
            decoder=self._node1_decoder) \
      .node(source=node2_path, node_type=self._node2_type,
            decoder=self._node2_decoder) \
      .edge(source=edge1_path,
            edge_type=(self._node1_type, self._node2_type, self._edge1_type),
            decoder=self._edge1_decoder, directed=False) \
      .edge(source=edge2_path,
            edge_type=(self._node2_type, self._node1_type, self._edge2_type),
            decoder=self._edge2_decoder) \
      .edge(source=edge3_path,
            edge_type=(self._node2_type, self._node2_type, self._edge3_type),
            decoder=self._edge3_decoder, directed=False)
    self.__class__.g.init(tracker=utils.TRACKER_PATH)

  @classmethod
  def setUpClass(cls):
    gl.set_default_neighbor_id(-1)
    gl.set_default_int_attribute(1000)
    gl.set_default_float_attribute(999.9)
    gl.set_default_string_attribute('hehe')
    gl.set_padding_mode(gl.REPLICATE) # default

    utils.prepare_env()

  @classmethod
  def tearDownClass(cls):
    if cls.g:
      cls.g.close()

  def setUp(self):
    """ prepare the data and the decoder.
    num_int_attrs = num_float_attrs = num_string_attrs = 2.
    dst_ids = utils.fixed_dst_ids(src_ids, dst_range).
    with fixed_dst_ids, the src_ids which src_id % 5 == 0 has no edge.
    """
    self._node1_type = "node1"
    self._node2_type = "node2"

    self._edge1_type = "edge1"
    self._edge2_type = "edge2"
    self._edge3_type = "edge3"

    self._node1_range = (0, 100)
    self._node2_range = (100, 200)

    self._node1_ids = range(self._node1_range[0], self._node1_range[1])
    self._node2_ids = utils.fixed_dst_ids(self._node1_ids, self._node2_range)

    self._node1_decoder = gl.Decoder(attr_types=utils.ATTR_TYPES)
    self._node2_decoder = gl.Decoder(weighted=True, labeled=True)
    self._edge1_decoder = gl.Decoder(attr_types=utils.ATTR_TYPES,
                                     labeled=True)
    self._edge2_decoder = gl.Decoder(attr_types=utils.ATTR_TYPES,
                                     weighted=True)
    self._edge3_decoder = gl.Decoder(weighted=True)

    self._seed_node1_ids = np.array([2, 7, 8])
    self._seed_node2_ids = np.array([102, 107, 108])

    self._seed_node1_ids_with_nbr_missing = np.array([5, 10, 110])
    self._seed_node2_ids_with_nbr_missing = np.array([102, 105, 108])
    # there has no edge whose src_id = 5 | 105

    self._default_dst_id = -1
    self._default_int_attr = 1000
    self._default_float_attr = 999.9
    self._default_string_attr = 'hehe'

    if self.needs_initial:
      self.initialize()
    if not self.g:
      time.sleep(1)

  def tearDown(self):
    pass
