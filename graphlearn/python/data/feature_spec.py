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
"""A FeatureSpec class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class SparseSpec(object):
  def __init__(self, bucket_size, dimension, need_hash):
    self.bucket_size = bucket_size
    self.dimension = dimension
    self.need_hash = need_hash

class DynamicSparseSpec(object):
  def __init__(self, dimension, need_hash):
    self.dimension = dimension
    self.need_hash = need_hash

class DenseSpec(object):
  def __init__(self):
    pass

class MultivalSpec(object):
  def __init__(self, bucket_size, dimension, delimiter):
    self.bucket_size = bucket_size
    self.dimension = dimension
    self.delimiter = delimiter

class DynamicMultivalSpec(object):
  def __init__(self, dimension, delimiter):
    self.dimension = dimension
    self.delimiter = delimiter

class FeatureSpec(object):
  """ Describes how to encode the values of `Nodes` or `Edges`.

  Args:
    feature_num (int, Required): The total feature count.
    weighted (boolean, Optional): Whether entity data has weight or not.
    labeled (boolean, Optional): Whether entity data has label or not.
  """
  def __init__(self, feature_num, weighted=False, labeled=False, timestamped=False):
    self._feature_num = feature_num
    self._weighted = weighted
    self._labeled = labeled
    self._timestamped = timestamped

    self._total_dim = 0

    self._int_spec_list = []
    self._float_spec_list = []
    self._string_spec_list = []

  @property
  def weighted(self):
    return self._weighted

  @property
  def labeled(self):
    return self._labeled

  @property
  def timestamped(self):
    return self._timestamped

  @property
  def int_specs(self):
    return self._int_spec_list

  @property
  def float_specs(self):
    return self._float_spec_list

  @property
  def string_specs(self):
    return self._string_spec_list

  @property
  def dimension(self):
    return self._total_dim

  def append_sparse(self, bucket_size, dimension, need_hash=False):
    if bucket_size is not None:
      self._int_spec_list.append(
        SparseSpec(bucket_size, dimension, need_hash))
    else:
      if need_hash:
        self._int_spec_list.append(
          DynamicSparseSpec(dimension, need_hash))
      else:
        self._string_spec_list.append(
          DynamicSparseSpec(dimension, need_hash))
    self._total_dim += dimension

  def append_dense(self, is_float=True):
    if is_float:
      self._float_spec_list.append(DenseSpec())
    else:
      self._int_spec_list.append(DenseSpec())
    self._total_dim += 1

  def append_multival(self, bucket_size, dimension, delimiter=","):
    """append description of multivalent feature."""
    if bucket_size is not None:
      self._string_spec_list.append(
        MultivalSpec(bucket_size, dimension, delimiter))
    else:
      self._string_spec_list.append(
        DynamicMultivalSpec(dimension, delimiter))
    self._total_dim += dimension
