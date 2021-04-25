/* Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef GRAPHLEARN_CORE_GRAPH_GRAPH_STORE_H_
#define GRAPHLEARN_CORE_GRAPH_GRAPH_STORE_H_

#include <vector>
#include <string>
#include "graphlearn/core/graph/graph.h"
#include "graphlearn/core/graph/heter_dispatcher.h"
#include "graphlearn/core/graph/noder.h"
#include "graphlearn/include/data_source.h"
#include "graphlearn/include/status.h"
#include "graphlearn/include/topology.h"

namespace graphlearn {

class Env;

class GraphStore {
public:
  explicit GraphStore(Env* env);
  ~GraphStore();

  static GraphStore* GetInstance();

  Status Init(const std::vector<io::EdgeSource>& edges,
              const std::vector<io::NodeSource>& nodes);
  Status Load(const std::vector<io::EdgeSource>& edges,
              const std::vector<io::NodeSource>& nodes);
  Status Build(const std::vector<io::EdgeSource>& edges,
               const std::vector<io::NodeSource>& nodes);

  Graph* GetGraph(const std::string& edge_type);
  Noder* GetNoder(const std::string& node_type);

  const Topology& GetTopology() const;

private:
  Env* env_;
  HeterDispatcher<Graph>* graphs_;
  HeterDispatcher<Noder>* noders_;
  Topology topo_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_GRAPH_GRAPH_STORE_H_
