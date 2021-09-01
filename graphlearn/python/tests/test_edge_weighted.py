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


class WeightedEdgeTestCase(EdgeTestCase):
  def test_weighted(self):
    file_path = self.gen_test_data([utils.WEIGHTED], False)
    decoder = gl.Decoder(weighted=True)
    g = gl.Graph() \
      .edge(source=file_path, edge_type=self.edge_tuple_, decoder=decoder)
    g.init(tracker=utils.TRACKER_PATH)

    query = g.E("first").batch(self.batch_size_).alias('e').values()
    ds = gl.Dataset(query, window=1)

    edges = ds.next()['e']
    utils.check_ids(edges.src_ids,
                    range(self.src_range_[0], self.src_range_[1]))
    utils.check_ids(edges.dst_ids,
                    range(self.dst_range_[0], self.dst_range_[1]))
    utils.check_edge_weights(edges)

    g.close()


if __name__ == "__main__":
  unittest.main()
