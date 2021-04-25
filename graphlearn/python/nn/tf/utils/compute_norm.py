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
'''Function for computing normalization for GraphSAINT.'''

import numpy as np

from graphlearn.python.errors import OutOfRangeError

np.seterr(divide='ignore')

def compute_norm(total_nodes,
                 total_edges,
                 subgraph_sampler,
                 sample_coverage=10):
  '''Function for computing node(loss) and edge(aggr) norm of GraphSAINT.
  Args:
    total_nodes: total nodes num.
    total_edges: total edges num.
    subgraph_sampler: SubGraphSampler instance.
    sample_coverage: sample coverage rate of nodes.
  Returns:
    node_norm: node normalization, a python dict.
    edge_norm: edge normalization, a numpy array.
  '''
  node_count = dict()
  edge_count = np.zeros(total_edges)
  edge_node_count = np.zeros(total_edges).astype(np.float32)
  node_norm = dict()

  total_sampled_subgraphs = 0
  total_sampled_nodes = 0

  while total_sampled_nodes < total_nodes * sample_coverage:
    while True:
      try:
        subgraph = subgraph_sampler.get()
        nodes = subgraph.nodes.ids
        edges = subgraph.edges.edge_ids
        for node_id in nodes:
          if not node_id in node_count:
            node_count[node_id] = 1
          else:
            node_count[node_id] += 1
        for i, edge_id in enumerate(edges):
          edge_count[edge_id] += 1
          # edges and nodes are in same order, so we get edge index
          # and figure out cooresponding row index to get node id.
          node_id = nodes[subgraph.row_indices[i]]
          edge_node_count[edge_id] += node_count[node_id]
        total_sampled_nodes += nodes.size
        total_sampled_subgraphs += 1
      except OutOfRangeError:
        break

  for k, v in node_count.iteritems():
    node_norm[k] = total_sampled_subgraphs * 1.0 / v / total_nodes

  edge_norm = np.clip(edge_node_count / edge_count, 0, 1e4)
  edge_norm[np.isnan(edge_norm)] = 0.1

  return node_norm, edge_norm
