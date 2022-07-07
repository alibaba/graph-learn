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
import numpy as np
import graphlearn.python.nn.tf as tfg

from graphlearn.python.nn.data import Data
from graphlearn.python.nn.hetero_subgraph import HeteroSubGraph



class HeteroEdgeInducer(tfg.SubGraphInducer):
  """ Induces the edge traversal and it's 1-hop query to HeteroSubGraph.
  """
  def __init__(self, use_neg=False, edge_types=None):
    super(HeteroEdgeInducer, self).__init__(use_neg=use_neg, 
      edge_types=edge_types)

  def induce_func(self, values):
    pos_src = values['pos_src']
    src_nbrs = values['src_hop1']
    pos_dst = values['pos_dst']
    dst_nbrs = values['dst_hop1']
    subgraphs = self.induce_hetero_graph(pos_src, pos_dst, src_nbrs, dst_nbrs)
    # negative samples.
    neg_subgraphs = None
    if self.use_neg:
      neg_dst = values['neg_dst']
      neg_dst_nbrs = values['neg_hop1']
      neg_subgraphs = self.induce_hetero_graph(pos_src, neg_dst, 
        src_nbrs, neg_dst_nbrs)
    return subgraphs, neg_subgraphs

  def induce_hetero_graph(self, src_nodes, dst_nodes, src_nbrs, dst_nbrs):
    """induce SubGraphs using edge and it's neighbors.
    Args:
      src_nodes: A gl.Nodes instance with shape [batch_size].
      dst_nodes: A gl.Nodes instance with shape [batch_size] or 
        [batch_size, 1] for negative sample.
      src_nbrs: The src_nodes' full neighbors with 1D shape.
      dst_nbrs: The dst_nodes' full neighbors with 1D shape.
    Returns:
      HeteroSubGraphs. For each HeteroSubGraph, nodes are concatenated 
      in the following order: for u: [src_node, dst_nbrs], for i: [dst_node, src_nbrs].
    """
    subgraphs = []
    u_data_list, ui_edge_index_list = self.induce_each_edge_type(src_nodes, src_nbrs, dst_nbrs)
    i_data_list, iu_edge_index_list = self.induce_each_edge_type(dst_nodes, dst_nbrs, src_nbrs)
    for i in range(len(u_data_list)):
      edge_index_dict = {('u', 'u-i', 'i'): ui_edge_index_list[i], 
                         ('i', 'u-i_reverse', 'u'): iu_edge_index_list[i]}
      nodes_dict = {'u': u_data_list[i], 'i': i_data_list[i]}
      subgraph = HeteroSubGraph(edge_index_dict, nodes_dict)
      subgraphs.append(subgraph)
    return subgraphs

  def induce_each_edge_type(self, src_nodes, src_nbrs, dst_nbrs):
    def _get_flatten_attrs(attrs, idx):
      if attrs is not None:
        return np.array([attrs[idx].reshape([-1])])
      return None
      
    dst_offset = 0
    data_list, edge_index_list = [], []
    for i in range (src_nodes.ids.size):
      # induce 1-hop enclosing SubGraph of target edge.
      src_id = src_nodes.ids[i].reshape([-1])
      int_attr = _get_flatten_attrs(src_nodes.int_attrs, i)
      float_attr = _get_flatten_attrs(src_nodes.float_attrs, i)
      string_attr = _get_flatten_attrs(src_nodes.float_attrs, i)
      row, col = [], []
      dst_begin, dst_end = dst_offset, dst_offset + dst_nbrs.offsets[i]
      ids, int_attrs, float_attrs, string_attrs = \
        self.concat_node_with_nbr(src_id, int_attr, float_attr, string_attr,
                                  dst_nbrs, dst_begin, dst_end)
      data_list.append(Data(ids, ints=int_attrs, floats=float_attrs, strings=string_attrs))
      row.append(0)
      col.append(0)
      for j in range(src_nbrs.offsets[i]): # src->dst
        row.append(0)
        col.append(j+1)
      edge_index_list.append(np.stack([np.array(row), np.array(col)], axis=0))
    return data_list, edge_index_list

  def concat_node_with_nbr(self, src_id, int_attrs, float_attrs, string_attrs,
                          nbr, begin, end):
    def concat_item(src, nbr):
      if src is not None:
        return np.concatenate((src, nbr), axis=0)
      return None
    ids = concat_item(src_id, nbr.ids[begin:end])
    if int_attrs is not None:
      int_attrs = concat_item(int_attrs, nbr.int_attrs[begin:end])
    if float_attrs is not None:
      float_attrs = concat_item(float_attrs, nbr.float_attrs[begin:end])
    if string_attrs is not None:
      string_attrs = concat_item(string_attrs, nbr.string_attrs[begin:end])
    return ids, int_attrs, float_attrs, string_attrs