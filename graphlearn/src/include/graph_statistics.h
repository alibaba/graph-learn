/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef GRAPHLEARN_INCLUDE_GRAPH_STATISTICS_H_
#define GRAPHLEARN_INCLUDE_GRAPH_STATISTICS_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace graphlearn {

using Counts = std::unordered_map<std::string, std::vector<int32_t>>;

class GraphStatistics {
public:
  GraphStatistics() = default;
  ~GraphStatistics() = default;

  const Counts& GetCounts() const;
  void AppendCount(const std::string& type, int32_t count);
  
private:
  // Number of nodes/edges of each type on each graph server.
  Counts counts_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_GRAPH_STATISTICS_H_
