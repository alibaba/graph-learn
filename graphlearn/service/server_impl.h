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

#ifndef GRAPHLEARN_SERVICE_SERVER_IMPL_H_
#define GRAPHLEARN_SERVICE_SERVER_IMPL_H_

#include <memory>
#include <string>
#include <vector>
#include "graphlearn/include/data_source.h"

namespace graphlearn {

class Env;
class Executor;
class GraphStore;
class InMemoryService;
class DistributeService;
class Coordinator;

class ServerImpl {
public:
  ServerImpl(int32_t server_id,
             int32_t server_count,
             const std::string& tracker);
  ~ServerImpl();

  void Start();
  void Init(const std::vector<io::EdgeSource>& edges,
            const std::vector<io::NodeSource>& nodes);
  void Stop();

private:
  void RegisterInMemoryService();
  void RegisterDistributeService();

private:
  int32_t server_id_;
  int32_t server_count_;
  Env*    env_;

  Executor*    executor_;
  GraphStore*  graph_store_;

  InMemoryService*   in_memory_service_;
  DistributeService* dist_service_;
  Coordinator*       coordinator_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_SERVER_IMPL_H_
