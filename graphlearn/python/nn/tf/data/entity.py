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

class Vertex(object):
  def __init__(self,
               ids=None,
               ints=None,
               floats=None,
               strings=None,
               labels=None,
               weights=None,
               out_degrees=None,
               in_degrees=None):
    """ ints, floats and strings are two dimensional, and the shapes are
      [int_attr_num, batch_size],
      [float_attr_num, batch_size],
      [string_attr_num, batch_size].
    out_degrees, in_degrees are tensors with shape of [batch_size], when the
    vertex has multiple types of downstream edges, then degrees are dicts of
    {edge_type: tensor}.
    """
    self.ids = ids
    self.int_attrs = ints
    self.float_attrs = floats
    self.string_attrs = strings
    self.labels = labels
    self.weights = weights
    self._out_degrees = out_degrees
    self._in_degrees = in_degrees

    self._handler_dict = {}

  def register_handler(self, key, value):
    self._handler_dict[key] = value

  @property
  def out_degrees(self):
    etype = self._handler_dict.get("out_edge_type")
    if etype:
      return self._out_degrees.get(etype)
    if isinstance(self._out_degrees, dict) and len(self._out_degrees) == 1:
      return self._out_degrees.values()[0]
    return self._out_degrees

  @property
  def in_degrees(self):
    # TODO(wenting.swt)
    return self._in_degrees
