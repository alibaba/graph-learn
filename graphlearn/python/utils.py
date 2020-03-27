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
"""This file contains util functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools


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
