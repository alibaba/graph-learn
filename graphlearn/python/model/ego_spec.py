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
"""Contains Describes class for EgoGraph and its elements."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class FeatureSpec(object):
  """ Describes gl.Nodes or gl.Edges, including the number of
  continuous features and categorical features, as well as whether
  the data is weighted or labeled.

  Args:
    cont_attrs_num: The number of continuous features.
    cate_attrs_num: The number of categorical features.
    labeled: Bool, set it to True if data contains labels
    weighted: Bool, set it to True if data contains weights
  """
  def __init__(self,
               cont_attrs_num,
               cate_attrs_num=0,
               labeled=False,
               weighted=False):

    self._cont_attrs_num = cont_attrs_num
    self._cate_attrs_num = cate_attrs_num
    self._labeled = labeled
    self._weighted = weighted

  @property
  def cont_attrs_num(self):
    return self._cont_attrs_num

  @property
  def cate_attrs_num(self):
    return self._cate_attrs_num

  @property
  def labeled(self):
    return self._labeled

  @property
  def weighted(self):
    return self._weighted


class HopSpec(object):
  """Describes 1 hop neighbors' Nodes and Edges.
    Args:
      node_spec: FeatureSpec of neighbor Nodes.
      edge_spec: FeatureSpec of neighbor Edges.
      sparse: Bool, set it to True If the neighbors in sparse format.
  """
  def __init__(self, node_spec, edge_spec=None, sparse=False):
    self._node_spec = node_spec
    self._edge_spec = edge_spec
    self._sparse = sparse

  @property
  def node_spec(self):
    return self._node_spec

  @property
  def edge_spec(self):
    return self._edge_spec

  @property
  def sparse(self):
    return self._sparse


class EgoSpec(object):
  """Describes a EgoGraph.
    Args:
    src_spec: FeatureSpec of root Nodes or Edges.
    hops_spec: A list of HopSpecs with length hops num.
  """
  def __init__(self, src_spec, hops_spec=None):
    self._src_spec = src_spec
    self._hops_spec = hops_spec

  @property
  def src_spec(self):
    return self._src_spec

  @property
  def hops_spec(self):
    return self._hops_spec
