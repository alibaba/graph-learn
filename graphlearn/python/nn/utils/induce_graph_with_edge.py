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
"""Functions of inducing SubGraph with target edge."""
import numpy as np

from graphlearn.python.nn.data import Data
from graphlearn.python.nn.subgraph import SubGraph

def induce_graph_with_edge(src_nodes, dst_nodes, 
  src_nbrs, dst_nbrs):
  """induce SubGraphs using edge and it's neighbors.
  Args:
    src_nodes: A gl.Nodes instance with shape [batch_size].
    dst_nodes: A gl.Nodes instance with shape [batch_size] or 
      [batch_size, 1] for negative sample.
    src_nbrs: The src_nodes' full neighbors with 1D shape.
    dst_nbrs: The dst_nodes' full neighbors with 1D shape.
  Returns:
    SubGraphs.
  """
  subgraphs = []
  src_offset, dst_offset = 0,0
  for i in range (src_nodes.ids.size):
    # induce k-hop enclosing SubGraph of target edge.
    ids = np.array([src_nodes.ids[i],
                    dst_nodes.ids[i].reshape([-1])]) # neg dst is 2D.
    int_attrs = _get_target_attrs(src_nodes.int_attrs, dst_nodes.int_attrs, i)
    float_attrs = _get_target_attrs(src_nodes.float_attrs, dst_nodes.float_attrs, i)
    string_attrs = _get_target_attrs(src_nodes.string_attrs, dst_nodes.string_attrs, i)

    row, col = [], []
    col_offset = ids.size
    src_begin, src_end = src_offset, src_offset + src_nbrs.offsets[i]
    dst_begin, dst_end = dst_offset, dst_offset + dst_nbrs.offsets[i]
    ids, int_attrs, float_attrs, string_attrs = \
      _concat_node_with_nbr(ids, int_attrs, float_attrs, string_attrs,
                            src_nbrs, src_begin, src_end)
    ids, int_attrs, float_attrs, string_attrs = \
      _concat_node_with_nbr(ids, int_attrs, float_attrs, string_attrs,
                            dst_nbrs, dst_begin, dst_end)
    row, col = gen_edge_index(i, src_nbrs, col_offset,
        row, col, src=True)
    col_offset += src_nbrs.offsets[i]
    row, col = gen_edge_index(i, dst_nbrs, col_offset,
        row, col, src=False)

    src_offset += src_nbrs.offsets[i]
    dst_offset += dst_nbrs.offsets[i]
    subgraph = SubGraph(np.stack([np.array(row),
                                  np.array(col)], axis=0),
                        Data(ids,
                             ints=int_attrs,
                             floats=float_attrs,
                             strings=string_attrs))
    subgraphs.append(subgraph)
  return subgraphs

def gen_edge_index(idx, src_nbrs, col_offset, 
                   row, col, src=True):
  """induce edge_index.
  """
  row_idx = 0 if src else 1
  for j in range(src_nbrs.offsets[idx]):
    row.append(row_idx)
    col.append(col_offset+j)
    col.append(row_idx)
    row.append(col_offset+j)
  return row, col

def _get_target_attrs(src_attrs, dst_attrs, index):
  if src_attrs is not None:
    return np.array([src_attrs[index], dst_attrs[index].reshape([-1])])
  return None

def _concat_node_with_nbr(ids, int_attrs, float_attrs, string_attrs,
                          nbr, begin, end):
  def concat_item(src, nbr):
    if src is not None:
      return np.concatenate((src, nbr), axis=0)
    return None
  ids = concat_item(ids, nbr.ids[begin:end])
  if int_attrs is not None:
    int_attrs = concat_item(int_attrs, nbr.int_attrs[begin:end])
  if float_attrs is not None:
    float_attrs = concat_item(float_attrs, nbr.float_attrs[begin:end])
  if string_attrs is not None:
    string_attrs = concat_item(string_attrs, nbr.string_attrs[begin:end])
  return ids, int_attrs, float_attrs, string_attrs