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
import graphlearn.python.tests.utils as utils
from graphlearn.python.tests.test_node_query import NodeQueryTestCase


class NodeQueryWeightTestCase(NodeQueryTestCase):
  def initialize(self):
    self.__class__.needs_initial = False
    file_path = self.gen_test_data([utils.ATTRIBUTED])
    decoder = gl.Decoder(attr_types=utils.ATTR_TYPES)
    self.__class__.g = gl.Graph() \
      .node(source=file_path,
            node_type=self.node_type_,
            decoder=decoder)
    self.__class__.g.init(tracker=utils.TRACKER_PATH)

  def test_query_exist(self):
    nodes = self.g.get_nodes(node_type=self.node_type_,
                             ids=self.exist_ids_)
    self.check_exist_attrs(nodes)

  def test_query_not_exist(self):
    nodes = self.g.get_nodes(node_type=self.node_type_,
                             ids=self.not_exist_ids_)
    self.check_not_exist_attrs(nodes)

  def test_query_exist_gremlin(self):
    nodes = self.g.V(self.node_type_, feed=self.exist_ids_) \
      .emit()
    self.check_exist_attrs(nodes)

  def test_query_not_exist_gremlin(self):
    nodes = self.g.V(self.node_type_, feed=self.not_exist_ids_) \
      .emit()
    self.check_not_exist_attrs(nodes)


if __name__ == "__main__":
  unittest.main()
