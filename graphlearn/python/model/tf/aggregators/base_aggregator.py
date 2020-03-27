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
"""Base class for aggregators"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod


class BaseAggregator(object):
  """Base class used to aggregate self and it's neighbor
  vectors.
  """
  __metaclass__ = ABCMeta

  @abstractmethod
  def aggregate(self, self_vecs, neigh_vecs):
    """

    Args:
      self_vecs: tensor, batch nodes' embedding vector, shape [B, D]
      neigh_vecs: tensor, corresponding neighbor nodes' embedding vector,
      shape [total_nbrs, D]

    Returns:
      aggregated nodes' embedding vector [B, H]
    """
    pass