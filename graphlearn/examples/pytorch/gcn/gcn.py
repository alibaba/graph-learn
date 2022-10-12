# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
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
"""GCN model based on PyG."""

import torch

import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
  def __init__(self,
               input_dim,
               hidden_dim,
               output_dim,
               depth=2,
               drop_rate=0.0,
               **kwargs):
    super(GCN, self).__init__()
    self.depth = depth
    self.drop_rate = drop_rate
    self.layers = torch.nn.ModuleList()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    for i in range(depth):
      input_dim = self.input_dim if i == 0 else self.hidden_dim
      output_dim = self.output_dim if i == self.depth - 1 else self.hidden_dim
      self.layers.append(GCNConv(input_dim, output_dim))

  def forward(self, data):
    """
    Args:
      data: PyG `Batch` object.

    Returns:
      output embedding.
    """
    h = data.x
    for l, layer in enumerate(self.layers):
      h = layer(h, data.edge_index)
      if l != self.depth - 1:
        h = F.relu(h)
        if self.drop_rate:
          h = F.dropout(h, p=self.drop_rate, training=self.training)
    return h

  def reset_parameters(self):
    for conv in self.layers:
        conv.reset_parameters()
