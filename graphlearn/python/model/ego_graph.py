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
"""Basic classes used to manage sampled subgraph."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graphlearn.python.model.utils.reorganize_attrs import reorganize_attrs
from graphlearn.python.values import Nodes

class EgoGraph(object):
  """This class organizes the sampled subgraph, which consists of a list of Layer,
  it starts from Nodes or Edges and uses hierarchical traversal order.

  Args:
    src: Nodes or Edges.
    layers: a list of Layer which contains neighbor Nodes and Edges(None in most case).
  """

  def __init__(self, src, layers):
    self._src = src
    self._hops = layers

  @property
  def src(self):
    return self._src

  @property
  def hops(self):
    return self._hops

  def flatten(self, spec):
    """Return flatten list format of EgoGraph.
    Note that all int_attrs and float_attrs from sampled results
    will be convert to continuous attrs.

    Args:
      spec: EgoGraphSpec
    """
    flatten_list = []
    # src, root Nodes or Edges
    flatten_list.extend(self._flatten_impl(self._src, spec.src_spec))

    if spec.hops_spec is None:
      return flatten_list

    assert len(self._hops) == len(spec.hops_spec), \
      'ego_spec num must be the same with hops num!'
    # neighbors
    for i in range(len(spec.hops_spec)):
      # Nodes
      if spec.hops_spec[i].node_spec is not None:
        flatten_list.extend(self._flatten_impl(
            self._hops[i].nodes,
            spec.hops_spec[i].node_spec,
            spec.hops_spec[i].sparse))
      # Edges
      if spec.hops_spec[i].edge_spec is not None:
        flatten_list.extend(self._flatten_impl(
            self._hops[i].edges,
            spec.hops_spec[i].edge_spec,
            spec.hops_spec[i].sparse))

    return flatten_list

  def _flatten_impl(self, feature, feature_spec=None, sparse=False):
    """help function for flatten.
    Args:
      feature: Nodes or Edges.
      feature_spec: A FeatureSpec object used to parse the feature.
      sparse: Set to True if the feature is in sparse format.
    """
    flatten_list = []
    # ids
    if isinstance(feature, Nodes):
      ids = feature.ids.reshape([-1])
    else: # Edges
      ids = feature.edge_ids.reshape([-1])
    flatten_list.append(ids)
    # for sparse format.
    if sparse:
      flatten_list.append(feature.offsets.reshape([-1]))
      flatten_list.append(feature.dense_shape)
      flatten_list.append(feature.indices)
    # labels
    if feature_spec.labeled:
      labels = feature.labels.reshape([-1])
      flatten_list.append(labels)
    # weights
    if feature_spec.weighted:
      weights = feature.weights.reshape([-1])
      flatten_list.append(weights)
    # attrs
    int_attrs = float_attrs = string_attrs = None
    if feature_spec.cont_attrs_num > 0:
      int_attrs = feature.int_attrs # [num_int_attrs, ids.shape]
      float_attrs = feature.float_attrs # [num_float_attrs,ids.shape]

    if feature_spec.cate_attrs_num > 0:
      string_attrs = feature.string_attrs
    continuous_attrs, categorical_attrs = \
      reorganize_attrs(int_attrs, float_attrs, string_attrs)

    if continuous_attrs is not None:
      flatten_list.append(continuous_attrs)

    if categorical_attrs is not None:
      flatten_list.append(categorical_attrs)

    return flatten_list
