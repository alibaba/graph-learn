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
""" Decoder is a user class which describes the data source schema.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Decoder(object):
  """ Decoder is used for Graph.node() and Graph.edge() to descripe
  the format of data source.
  """

  def __init__(self,
               weighted=False,
               labeled=False,
               attr_types=None,
               attr_delimiter=":"):
    """ Data source decoder.
    Args:
      weighted (boolean, Optional): Indicates whether or not
        the data source has weights. Default is False.
      labeled (boolean, Optional): Indicates whether or not
        the data source has labels. Default is False.
      attr_types (list, Optional): Indicates each column type of attributes
        if attributes exists. Default is None, it means no attribute exist.
      attr_delimiter (string, Optional): Indicates the delimiter that seperate
        each attribute. Default is ':'.
    """
    self._weighted = weighted
    self._attributed = False
    self._attr_types = attr_types
    self._attr_delimiter = attr_delimiter
    self._labeled = labeled

    self._int_attr_num = 0
    self._float_attr_num = 0
    self._string_attr_num = 0
    attr_mapping_index = [[], [], []]
    if self.attr_types:
      if not isinstance(self.attr_types, list):
        raise ValueError("attr_types for Decoder must be a list, got {}."
                         .format(type(self._attr_types)))
      self._attributed = True
      for i in range(len(self._attr_types)):
        hash_to_int = False
        t = self._attr_types[i]
        if isinstance(t, tuple):
          type_name = t[0]
          if len(t) >= 2:
            hash_to_int = True
        else:
          type_name = t
        self._int_attr_num += int(type_name == "int") + \
            int(type_name == "string" and hash_to_int)
        self._float_attr_num += int(type_name == "float")
        self._string_attr_num += int(type_name == "string" and not hash_to_int)

  @property
  def weighted(self):
    return self._weighted

  @property
  def labeled(self):
    return self._labeled

  @property
  def attributed(self):
    return self._attributed

  @property
  def attr_types(self):
    return self._attr_types

  @property
  def attr_delimiter(self):
    return self._attr_delimiter

  @property
  def data_format(self):
    return int(self._weighted * 2 + \
               self._labeled * 4 + self._attributed * 8)

  @property
  def int_attr_num(self):
    return self._int_attr_num

  @property
  def float_attr_num(self):
    return self._float_attr_num

  @property
  def string_attr_num(self):
    return self._string_attr_num

  def format_attrs(self, int_attrs, float_attrs, string_attrs):
    """ Reshape and format attributes with int_attr_num, float_attr_num
    and string_attr_num calculated by decoder.attr_types.

    Return:
      Reshaped int_attrs, float_attrs, string_attrs
    """
    if int_attrs is not None:
      int_attrs = int_attrs.reshape(-1, self._int_attr_num)

    if float_attrs is not None:
      float_attrs = float_attrs.reshape(-1, self._float_attr_num)

    if string_attrs is not None:
      string_attrs = np.array([s.decode('utf-8', errors='ignore') for s in string_attrs])
      string_attrs = string_attrs.reshape(-1, self._string_attr_num)
      string_attrs = string_attrs.astype('U13')
    return int_attrs, float_attrs, string_attrs
