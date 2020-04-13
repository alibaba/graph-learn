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
"""Classes used to construct TensorFlow'tensor format of EgoGraph."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class FeatureTensor(object):
  """TensorFlow Tensor format of Nodes or Edges, including tensors of ids,
  continuous features and categorical features. For the weighted or labeled
  entity data, weights tensor and label tensor can be accessed. Both dense
  and sparse type of data are supported.
  """
  def __init__(self,
               ids,
               continuous_attrs=None,
               categorical_attrs=None,
               labels=None,
               weights=None,
               offsets=None,
               dense_shape=None,
               indices=None):
    """
    Args:
      ids: A 1D tensor, a batch of ids of nodes or edges.
      continuous_attrs: A tensor of continuous attributes with shape
      [ids.shape, num_continuous].
      categorical_attrs: A tensor of categorical attributes with shape
      [ids.shape, num_continuous].
      labels: A tensor of labels with shape [ids.shape].
      weights: A tensor of weights with shape [ids.shape].
      offsets: A 1D tensor recording the number of values on each row
      for a sparse tensor.
      dense_shape: A 1D tensor recording the corresponding 2D dense shape
      for a sparse tensor.
      indices: A 1D tensor recording the indices of the elements for a sparse
      tensor that contains nonzero values.
    """
    self._ids = ids
    self._continuous_attrs = continuous_attrs
    self._categorical_attrs = categorical_attrs
    self._labels = labels
    self._weights = weights
    self._offsets = offsets
    self._dense_shape = dense_shape
    self._indices = indices

  @property
  def ids(self):
    return self._ids

  @property
  def continuous_attrs(self):
    return self._continuous_attrs

  @property
  def categorical_attrs(self):
    return self._categorical_attrs

  @property
  def weights(self):
    return self._weights

  @property
  def labels(self):
    return self._labels

  @property
  def offsets(self):
    return self._offsets

  @property
  def dense_shape(self):
    return self._dense_shape

  @property
  def indices(self):
    return self._indices


class HopTensor(object):
  """Tensor of 1 hop neighbors' Nodes and Edges.

  Args:
    node_tensor: Node FeatureTensor.
    edge_tensor: Edge FeatureTensor.
  """
  def __init__(self, node_tensor, edge_tensor=None):
    self._node_tensor = node_tensor
    self._edge_tensor = edge_tensor

  @property
  def nodes(self):
    return self._node_tensor

  @property
  def edges(self):
    return self._edge_tensor


class EgoTensor(object):
  """Tensor format of EgoGraph.

  Args:
    tensors: A tuple of tensors corresponding to EgoGraph flatten format.
    spec: An EgoSpec object used to parse the tensors.
  """
  def __init__(self, tensors, spec):
    self._tensors = tensors
    self._spec = spec
    self._idx = 0

    self._src, self._hops = self._build()

  @property
  def src(self):
    return self._src

  @property
  def hops(self):
    return self._hops

  def _next(self):
    t = self._tensors[self._idx]
    self._idx += 1
    return t

  def _build(self):
    """Build EgoGraphTensor.

    Returns:
      src FeatureTensor and neighbors HopTensor list.
    """
    # src
    src_tensor = self._build_impl(self._spec.src_spec)
    # neighbors
    nbr_tensor_list = [] # for each hop's neighbors.

    if self._spec.hops_spec is None:
      return src_tensor, nbr_tensor_list

    for i in range(len(self._spec.hops_spec)):
      # Nodes
      if self._spec.hops_spec[i].node_spec is not None:
        node_tensor = self._build_impl(
            self._spec.hops_spec[i].node_spec,
            self._spec.hops_spec[i].sparse)
      else:
        node_tensor = None
      # Edges
      if self._spec.hops_spec[i].edge_spec is not None:
        edge_tensor = self._build_impl(
            self._spec.hops_spec[i].edge_spec,
            self._spec.hops_spec[i].sparse)
      else:
        edge_tensor = None
      nbr_tensor_list.append(HopTensor(node_tensor, edge_tensor))

    return src_tensor, nbr_tensor_list

  def _build_impl(self, feature_spec, sparse=False):
    """Constructs FeatureTensor.

    Args:
      feature_spec: A FeatureSpec object used to parse the feature.
      sparse: Set to True if the feature is in sparse format.
    Returns:
      FeatureTensor.
    """
    ids = self._next()
    if sparse:
      offsets = self._next()
      dense_shape = self._next()
      indices = self._next()
    else:
      offsets = None
      dense_shape = None
      indices = None

    if feature_spec.labeled:
      labels = self._next()
    else:
      labels = None

    if feature_spec.weighted:
      weights = self._next()
    else:
      weights = None

    if feature_spec.cont_attrs_num > 0:
      continuous_attrs = self._next()
    else:
      continuous_attrs = None

    if feature_spec.cate_attrs_num > 0:
      categorical_attrs = self._next()
    else:
      categorical_attrs = None

    feature_tensor = FeatureTensor(ids,
                                   continuous_attrs=continuous_attrs,
                                   categorical_attrs=categorical_attrs,
                                   labels=labels,
                                   weights=weights,
                                   offsets=offsets,
                                   dense_shape=dense_shape,
                                   indices=indices)
    return feature_tensor
