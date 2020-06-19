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
from graphlearn.python.tests.test_edge import EdgeTestCase


class AttributedEdgeTestCase(EdgeTestCase):
  def test_attributed(self):
    file_path = self.gen_test_data([utils.ATTRIBUTED], False)
    decoder = gl.Decoder(attr_types=utils.ATTR_TYPES)
    g = gl.Graph() \
      .edge(source=file_path, edge_type=self.edge_tuple_, decoder=decoder)
    g.init(tracker=utils.TRACKER_PATH)

    edges = g.get_edges(edge_type="first",
                        src_ids=self.src_ids_,
                        dst_ids=self.dst_ids_)

    g.close()


if __name__ == "__main__":
  unittest.main()
