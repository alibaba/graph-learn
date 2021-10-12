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

import numpy as np
from graphlearn.python.nn.data import Data


class SubGraph(object):
  """ `SubGraph` is a basic data structure used to describe a sampled 
  subgraph. It constists of `edge_index` and nodes `Data` and edges `Data`.

  Args:
    edge_index: A np.ndarray object with shape [2, batch_size], which indicates
      [rows, cols] of SubGraph.
    nodes: A `Data`/ndarray object denoting the input nodes.
    edges: A `Data`/ndarray object denoting the input edges.
    
  Note that this object can be extented by any other additional data.
  """
  def __init__(self, edge_index, nodes, edges=None, **kwargs):
    self._edge_index = edge_index
    self._nodes = nodes
    self._edges = edges
    for key, item in kwargs.items():
      self[key] = item

  @property
  def num_nodes(self):
    if isinstance(self._nodes, Data):
      return self._nodes.ids.size
    else:
      return self._nodes.size

  @property
  def num_edges(self):
    return self._edge_index.shape[1]

  @property
  def nodes(self):
    return self._nodes

  @property
  def edge_index(self):
    return self._edge_index

  @property
  def edges(self):
    return self._edges

  @property
  def keys(self):
    r"""Returns all names of graph attributes."""
    keys = [key for key in self.__dict__.keys() if self[key] is not None]
    keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
    return keys

  def __getitem__(self, key):
    return getattr(self, key, None)

  def __setitem__(self, key, value):
    setattr(self, key, value)