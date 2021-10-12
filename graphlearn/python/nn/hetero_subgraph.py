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


class HeteroSubGraph(object):
  """ A data object describing heterogeneous `SubGraph`.
  Different types of nodes and edges are represented by a dict.

  Args:
    edge_index_dict: A dict of np.ndarray objects. Each key is a tuple of 
      (src_type, edge_type, dst_type) and each value indicates [rows, cols].
    nodes_dict: A dict of `Data`/ndarray object denoting different types of nodes.
    edges_dict: A dict of `Data`/ndarray object denoting different types of edges.
  
  Examples:
    For meta-path "user-click-item", the HeteroSubGraph may be created as follows:
      edge_index_dict[('user', 'click', 'item')] = np.array([[0,1,2], [2,3,4]])
      edges_dict[('user', 'click', 'item')] = Data(...)
      nodes_dict['user'] = Data(...)
      nodes_dict['item'] = Data(...)
      hg = HeteroSubGraph(edge_index_dict, nodes_dict, edges_dict)

  """
  def __init__(self, edge_index_dict, nodes_dict, edges_dict=None, **kwargs):
    self._edge_index_dict = edge_index_dict
    self._nodes_dict = nodes_dict
    self._edges_dict = edges_dict
    for key, item in kwargs.items():
      self[key] = item

  def num_nodes(self, node_type):
    if isinstance(self._nodes_dict[node_type], Data):
      return self._nodes_dict[node_type].ids.size
    else:
      return self._nodes_dict[node_type].size

  def num_edges(self, edge_type):
    return self._edge_index_dict[edge_type].shape[1]

  @property
  def nodes_dict(self):
    return self._nodes_dict

  @property
  def edge_index_dict(self):
    return self._edge_index_dict

  @property
  def edges_dict(self):
    return self._edges_dict

  @property
  def keys(self):
    r"""Returns all names of graph attributes."""
    keys = [key for key in self.__dict__.keys() if self[key] is not None]
    keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
    return keys

  @property
  def node_types(self):
    """Returns all node types of the heterogeneous subgraph."""
    return list(self._nodes_dict.keys())

  @property
  def edge_types(self):
    """Returns all edge types of the heterogeneous subgraph."""
    return list(self._edge_index_dict.keys())

  def __getitem__(self, key):
    return getattr(self, key, None)

  def __setitem__(self, key, value):
    setattr(self, key, value)