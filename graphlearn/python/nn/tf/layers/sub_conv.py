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
"""Abstract class of SubGraph based graph convolutional layer"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
from graphlearn.python.nn.tf.module import Module


class SubConv(Module):
  __metaclass__ = ABCMeta

  @abstractmethod
  def forward(self, edge_index, node_vec, **kwargs):
    """
    Args:
      edge_index: A Tensor. Edge index of subgraph.
      node_vec: A Tensor or a list of two Tensors(for heterogeneous graph) 
      which means node embeddings with shape [batchsize, dim].
    Returns:
      A tensor. output embedding with shape [batch_size, output_dim].
    """
    pass