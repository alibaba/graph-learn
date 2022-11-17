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

import sys
import numpy as np

from graphlearn.python.data.feature_spec import FeatureSpec

class Decoder(object):
  """ Decoder is used for Graph.node() and Graph.edge() to describe the schema
      of data source.
  """

  def __init__(self,
               weighted=False,
               labeled=False,
               attr_types=[],
               attr_delimiter=":",
               attr_dims=[]):
    """ Initialize a data source decoder.

    Args:
      weighted (boolean, Optional): Whether the data source has weights.
        Default is False.
      labeled (boolean, Optional): Whether the data source has labels.
        Default is False.
      attr_types (list, Optional): Attribute type list if attributes exist.
        Default is None, which means no attribute exists. Each element must 
        be string, tuple or list.

        Valid types like below:
        attr_types = ['int', 'float', 'string']
        attr_types = ['int', ('string', 10), 'string'] # 10 means bucket size
        For ('string', 10), we will get the string and hash it into 10 buckets
        directly. The raw attribute can by any string.

        # True means multi-val splited by ',', only for string attribute
        attr_types = ['int', ('string', 10, True)]

        # 10 means bucket size
        attr_types = ['int', ('int', 10), 'string']
        For ('int', 10), we will cast the string to int first and then hash it
        into 10 buckets. In this way, the raw attribute must be an integer.

        When `attr_dims` is assigned, be sure that the string attribute must
        be configured with a bucket size. Bucket size for int attribute is
        optional, and it will be considered as a continuous attribute if bucket
        size is not assigned.

      attr_delimiter (string, Optional): The delimiter to seperate attributes.
        Default is ':'. If attributes exist, all of them are concatenated
        together with a delimiter in the raw storage. We need to know how to
        parse them.
      attr_dims (list, Optional): An integer list, the element of which
        represents the dimension of the corresponding attribute that will be
        encodeding to. Default is None, which means no attribute exists.

        All valid configurations of attr_type and attr_dim are as shown below.
        |    attr_type     |attr_dim|             encoded into                 |
        |    ---------     | -- |                  --------                    |
        |     "string"     | 8  |   Dynamic bucket embedding variable, dim=8   |
        |   ("string",10)  | 8  |   Embedding variable, bucketsize=10, dim=8   |
        |("string",10,True)| 8  |Sparse embedding variable, bucketsize=10,dim=8|
        |("string",None,True)| 8|   Sparse dynamic embedding variable, dim=8   |
        |       "int"      |None|           Continues numeric tensor           |
        |       "int"      | 8  |   Dynamic bucket embedding variable, dim=8   |
        |    ("int",10)    | 8  |   Embedding variable, bucket size=10, dim=8  |
        |      "float"     |None|           Continues numeric tensor           |
        Note that dynamic bucket embedding variable is only supported in PAI-TF.
        For continues numeric attribute, attr_dim should be either None or 0.
    """
    self._weighted = weighted
    self._labeled = labeled
    self._attr_types = attr_types
    self._attr_delimiter = attr_delimiter
    self._attr_dims = attr_dims

    self._int_attr_num = 0
    self._float_attr_num = 0
    self._string_attr_num = 0
    self._fspec = None

    self._attributed = self._parse_attributes()

  def _parse_attributes(self):
    if not self._attr_types:
      return False
    if not isinstance(self._attr_types, list):
      raise ValueError("attr_types for Decoder must be a list, got {}."
                       .format(type(self._attr_types)))
    for i in range(len(self._attr_types)):
      type_name, bucket_size, is_multival = self.parse(self._attr_types[i])
      self._int_attr_num += int(type_name == "int")
      self._float_attr_num += int(type_name == "float")
      if is_multival:
        self._string_attr_num += 1
      else:
        self._int_attr_num += int(type_name == "string" and bucket_size is not None)
        self._string_attr_num += int(type_name == "string" and bucket_size is None)
    return True

  def _build_feature_spec(self):
    num_attrs = len(self._attr_types)
    numeric_types = ("float", "int")
    embedding_types = ("int", "string")

    self._fspec = FeatureSpec(num_attrs, self._weighted, self._labeled)

    if not self._attr_dims:
      self._attr_dims = [None for _ in range(num_attrs)]

    if num_attrs != len(self._attr_dims):
      raise ValueError("The size of attr_dims must be equal with attr_types.")

    def check(dim, attr_type, bucket):
      if not dim:
        assert type_name in numeric_types and bucket_size is None, \
          "Must assign an attr_dim for {}, and bucket_size should None." \
          .format(type_name)
      else:
        assert type_name in embedding_types, \
          "Must assign an attr_dim with None for {}".format(type_name)

    for attr_type, dim in zip(self._attr_types, self._attr_dims):
      type_name, bucket_size, is_multival = self.parse(attr_type)
      check(dim, type_name, bucket_size)
      if is_multival:
        self._fspec.append_multival(bucket_size, dim, ",")
      elif dim:
        self._fspec.append_sparse(bucket_size, dim, type_name == "int")
      else:
        self._fspec.append_dense(type_name == "float")

  def parse(self, attr_type):
    if isinstance(attr_type, tuple) or isinstance(attr_type, list):
      type_name = attr_type[0]
      bucket_size = attr_type[1] if len(attr_type) >= 2 else None
      is_multival = attr_type[2] if len(attr_type) >= 3 else False
    else:
      type_name = attr_type
      bucket_size = None
      is_multival = False

    assert type_name in {"int", "float", "string"}

    if is_multival and type_name != "string":
      raise ValueError("multi-value attribute must be string type.")
    return type_name, bucket_size, is_multival

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
    # attributed << 3 | labeled << 2 | weighted << 1
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

  @property
  def feature_spec(self):
    if not self._fspec:
      self._build_feature_spec()
    return self._fspec

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
      if (sys.version_info[0] == '3'):
        # For Python 3.X, encode string attributes as Unicode.
        string_attrs = string_attrs.astype('U')

    return int_attrs, float_attrs, string_attrs
