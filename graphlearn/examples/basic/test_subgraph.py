# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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

import getopt
import os
import sys

import graphlearn as gl
from query_examples import *


def main(argv):
  cur_path = sys.path[0]
  g = gl.Graph()
  g.node(os.path.join(cur_path, "data/user"),
         node_type="user", decoder=gl.Decoder(weighted=True)) \
    .node(os.path.join(cur_path, "data/entity"),
          node_type="entity", decoder=gl.Decoder(attr_types=['float', 'float', 'float', 'float'], labeled=True)) \
    .edge(os.path.join(cur_path, "data/relation"),
          edge_type=("entity", "entity", "relation"), decoder=gl.Decoder(weighted=True), directed=False)

  g.init()
  num_nbrs = [10, 10]
  edge_sampler = g.edge_sampler('relation', batch_size=1, strategy="by_order") # bz must be 1.
  node_sampler = g.node_sampler('entity', batch_size=10, strategy="by_order")
  sampler = g.subgraph_sampler('relation', num_nbrs, need_dist=True) # random_edge, in_order_edge
  edges = edge_sampler.get()
  nodes = node_sampler.get()
  #subgraph = sampler.get(nodes.ids)
  subgraph = sampler.get(edges.src_ids, edges.dst_ids)
  print('edge_index: ', subgraph.edge_index)
  print('nodes: ', subgraph.nodes.ids)
  print('edges: ', subgraph.edges.edge_ids)
  print("dist_to_src :", subgraph.dist_to_src)
  print("dist_to_dst :", subgraph.dist_to_dst)

  num_nbrs=[2]
  # test GSL.
  query = g.E('relation').batch(1).shuffle().alias('relation').SubGraph('relation', num_nbrs, need_dist=True).alias('sub').values()
  #query = g.V('entity').shuffle().batch(128).SubGraph('relation', num_nbrs, need_dist=False).alias('sub').values()
  ds = gl.Dataset(query)
  subgraph = ds.next()['sub']
  print('edge_index: ', subgraph.edge_index)
  print('nodes: ', subgraph.nodes.ids)
  print('nodes.float_attrs: ', subgraph.nodes.float_attrs)
  print('edges: ', subgraph.edges.edge_ids)
  print("dist_to_src :", subgraph.dist_to_src)
  print("dist_to_dst :", subgraph.dist_to_dst)
  g.close()


if __name__ == "__main__":
  main(sys.argv[1:])
