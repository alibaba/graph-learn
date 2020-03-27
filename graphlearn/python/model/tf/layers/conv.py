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
"""Abstract class of graph convolutional layer"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod


class BaseConv(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def forward(self, self_vecs, neigh_vecs, segment_ids=None):
    """
    Args:
      self_vecs: A Tensor. Self raw feature embeddings with shape
      [batchsize, dim].
      neigh_vecs: A Tensor. Neighbor raw feature embeddings with shape
      [batch_size, nbr_num, dim] in dense input, or [total_nbr_num, dim]
      in sparse input.
      segment_ids: A Tensor. Used for sparse input mode. Whose element
      indicates the index of its root nodes.
    Returns:
      A tensor. output embedding with shape [batch_size, output_dim].
    """
    pass
