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

#ifndef GRAPHLEARN_CORE_OPERATOR_SUBGRAPH_SUBGRAPH_UTILS_H_
#define GRAPHLEARN_CORE_OPERATOR_SUBGRAPH_SUBGRAPH_UTILS_H_

#include <cstdint>
#include <vector>

namespace graphlearn {
namespace op {
namespace subgraph {

class Graph {
public:
  Graph(int32_t num_nodes);
  void AddEdge(int32_t row, int32_t col);
  std::vector<int32_t> BFSShortestPath(int32_t s);

private:
  int32_t num_nodes_;
  std::vector<std::vector<int32_t>> adj_;
};

}  // namespace subgraph
}  // namespace op
}  // namespace graphlearn

# endif  // GRAPHLEARN_CORE_OPERATOR_SUBGRAPH_SUBGRAPH_UTILS_H_