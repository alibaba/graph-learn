/* Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/operator/subgraph/subgraph_utils.h"

#include <climits>
#include <queue>

namespace graphlearn {
namespace op {
namespace subgraph {

Graph::Graph(int32_t num_nodes) {
  num_nodes_ = num_nodes;
  adj_.resize(num_nodes);
  for (auto& row : adj_) {
    row.reserve(num_nodes);
  }
}

void Graph::AddEdge(int32_t row, int32_t col) {
  adj_[row].push_back(col);
}

std::vector<int32_t> Graph::BFSShortestPath(int32_t s) {
  std::vector<int32_t> dist;
  std::vector<bool> visited;
  dist.resize(num_nodes_, INT32_MAX);
  visited.resize(num_nodes_, false);
  std::queue<int32_t> q;

  visited[s] = true;
  q.push(s);
  dist[s] = 0;
  while (!q.empty()) {
    s = q.front();
    q.pop();
    for (auto nbr : adj_[s]) {
      if (!visited[nbr]) {
        visited[nbr] = true;
        dist[nbr] = dist[s] + 1;  // parent_node_distance + 1
        q.push(nbr);
      }
    }
  }
  return dist;
}

}  // namespace subgraph
}  // namespace op
}  // namespace graphlearn