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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Data(object):
  """A plain object modeling a batch of `Nodes` or `Edges`."""
  def __init__(self,
               ids=None,
               ints=None,
               floats=None,
               strings=None,
               labels=None,
               weights=None,
               dst_ids=None,
               offsets=None,
               indices=None,
               dense_shape=None,
               **kwargs):
    """ ints, floats and strings are attributes in numpy or Tensor format 
      with the shape
      [batch_size, int_attr_num],
      [batch_size, float_attr_num],
      [batch_size, string_attr_num].

      labels and weights are in numpy or Tensor format with the shape 
      [batch_size], [batch_size].

      dst_ids is used for Edges/SparseEdges.

      offsets, indices and dense_shpae is used for SpareNodes/SparseEdges.

      The data object can be extented by any other additional data.
    """
    self.ids = ids
    self.int_attrs = ints
    self.float_attrs = floats
    self.string_attrs = strings
    self.labels = labels
    self.weights = weights
    self.dst_ids = dst_ids    
    self.offsets = offsets
    self.indices = indices
    self.dense_shape = dense_shape
    for key, item in kwargs.items():
      self[key] = item

    self._handler_dict = {}

  def apply(self, func):
    """Applies the function `func` to all attributes.
    """
    for k, v in self.__dict__.items():
      if v is not None and k[:2] != '__' and k[-2:] != '__':
        self.__dict__[k] = func(v)
    return self

  def __getitem__(self, key):
    return getattr(self, key, None)

  def __setitem__(self, key, value):
    setattr(self, key, value)