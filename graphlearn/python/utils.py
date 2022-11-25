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

import functools
from enum import Enum

def deprecated(date, old, instead):
  """A decorator to print log of api change.
  """
  def log_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      print('[WARNING] %s will not be supported after %s,'
            ' please use %s instead.' % (old, date, instead))
      return func(*args, **kwargs)
    return wrapper
  return log_decorator

def strategy2op(strategy, op_type):
  """ Convert strategy to op.
  """
  op = ''
  if isinstance(strategy, str):
    for s in strategy.split('_'):
      op += s.capitalize()
  op += op_type
  return op

class Mask(Enum):
  NONE = 0
  TRAIN = 1
  TEST = 2
  VAL = 3

def get_mask_type(raw_type, mask=Mask.NONE):
  """ Get the masked type for raw node_type or edge_type.
  For NONE mask, return the raw_type.
  TRAIN mask for raw_type of "user", return "MASK*user". 
  TEST mask for raw_type of "user, return "MASK**user".
  VAL mask for raw_type of "user, return "MASK***user".
  """
  assert isinstance(raw_type, str)
  if isinstance(mask, str):  # accept string type
    mask = Mask[mask.upper()]
  assert isinstance(mask, Mask)
  if mask == Mask.NONE:
    return raw_type
  return "MASK" + "*" * mask.value + raw_type
