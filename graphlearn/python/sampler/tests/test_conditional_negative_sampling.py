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

import numpy as np
import unittest

from graphlearn.python.sampler.tests.test_sampling import SamplingTestCase
import graphlearn.python.tests.utils as utils

def _check_ids(pos_id, neg_ids):
  utils.check_val_equal(neg_ids[0] % 5, pos_id % 5)
  utils.check_val_equal(neg_ids[1] % 4, pos_id % 4)
  utils.check_val_equal(neg_ids[2] % 3, pos_id % 3)
  utils.check_val_equal(neg_ids[3] % 3, pos_id % 3)


class ConditionalNegavtiveSamplingTestCase(SamplingTestCase):
  """ Test cases for conditional negative sampler.
  """
  def test_indegree(self):
    """ Sample negative neighbors with in-degree of the target nodes.
    """
    expand_factor = 4
    src_ids = np.array([1,2,3,4,5])
    dst_ids = np.array([12,34,2,67,128])
    nbr_s = self.g.negative_sampler(self._cond_edge_type,
                                    expand_factor=expand_factor,
                                    strategy="in_degree",
                                    conditional=True,
                                    unique=False,
                                    batch_share=False,
                                    int_cols=[0,1],
                                    int_props=[0.25,0.25],
                                    str_cols=[0],
                                    str_props=[0.5])
    nodes = nbr_s.get(src_ids, dst_ids)

    for idx, id in enumerate(src_ids):
      print('src_id:', id, 'dst_id:', dst_ids[idx], 'neg_ids:', nodes.ids[idx])
      nbr_ids = [id+2,id+3,id+5]
      utils.check_disjoint(nodes.ids[idx], nbr_ids)
      _check_ids(dst_ids[idx], nodes.ids[idx])

  def test_random(self):
    """ Sample negative neighbors randomly of the target nodes.
    """
    expand_factor = 4
    src_ids = np.array([1,2,3,4,5])
    dst_ids = np.array([12,34,2,67,128])
    nbr_s = self.g.negative_sampler(self._cond_edge_type,
                                    expand_factor=expand_factor,
                                    strategy="random",
                                    conditional=True,
                                    unique=False,
                                    batch_share=False,
                                    int_cols=[0,1],
                                    int_props=[0.25,0.25],
                                    str_cols=[0],
                                    str_props=[0.5])
    nodes = nbr_s.get(src_ids, dst_ids)

    for idx, id in enumerate(src_ids):
      print('src_id:', id, 'dst_id:', dst_ids[idx], 'neg_ids:', nodes.ids[idx])
      nbr_ids = [id+2,id+3,id+5]
      utils.check_disjoint(nodes.ids[idx], nbr_ids)
      _check_ids(dst_ids[idx], nodes.ids[idx])

  def test_node_weight(self):
    """ Sample negative neighbors with node weight.
    """
    expand_factor = 4
    src_ids = np.array([1,2,3,4,5])
    dst_ids = np.array([12,34,2,67,128])
    nbr_s = self.g.negative_sampler(self._cond_node_type,
                                    expand_factor=expand_factor,
                                    strategy="node_weight",
                                    conditional=True,
                                    unique=False,
                                    batch_share=True,
                                    int_cols=[0,1],
                                    int_props=[0.25,0.25],
                                    str_cols=[0],
                                    str_props=[0.5])
    nodes = nbr_s.get(src_ids, dst_ids)

    for idx, id in enumerate(src_ids):
      print('src_id:', id, 'dst_id:', dst_ids[idx], 'neg_ids:', nodes.ids[idx])
      utils.check_disjoint(nodes.ids[idx], dst_ids)
      _check_ids(dst_ids[idx], nodes.ids[idx])

if __name__ == "__main__":
  unittest.main()
