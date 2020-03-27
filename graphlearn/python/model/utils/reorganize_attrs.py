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
"""Reorganize attributes"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def reorganize_attrs(int_attrs, float_attrs, string_attrs):
  """Reformats attributes."""
  def reshape_attrs(attrs):
    """reshape input attrs to [batch_size, attr_num].
    """
    if attrs is not None:
      return np.reshape(attrs, (-1, attrs.shape[-1]))
    return attrs

  int_attrs = reshape_attrs(int_attrs)
  float_attrs = reshape_attrs(float_attrs)
  string_attrs = reshape_attrs(string_attrs)

  continuous_list = []
  if int_attrs is not None:
    continuous_list.append(int_attrs.astype(np.float32))
  if float_attrs is not None:
    continuous_list.append(float_attrs)

  if continuous_list:
    continuous_attrs = np.concatenate(continuous_list, axis=-1)
  else:
    continuous_attrs = None
  if string_attrs is not None:
    categorical_attrs = string_attrs
  else:
    categorical_attrs = None
  return continuous_attrs, categorical_attrs
