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

#ifndef GRAPHLEARN_INCLUDE_CLIENT_H_
#define GRAPHLEARN_INCLUDE_CLIENT_H_

#include "include/aggregating_request.h"
#include "include/constants.h"
#include "include/dag_request.h"
#include "include/graph_request.h"
#include "include/random_walk_request.h"
#include "include/sampling_request.h"
#include "include/status.h"
#include "include/subgraph_request.h"

namespace graphlearn {

class ClientImpl;
class StateRequestPb;

class Client {
public:
  ~Client();

#define DECLARE_METHOD(Name)                \
  Status Name(const Name##Request* request, \
              Name##Response* response)

  DECLARE_METHOD(UpdateEdges);
  DECLARE_METHOD(UpdateNodes);
  DECLARE_METHOD(GetEdges);
  DECLARE_METHOD(LookupEdges);
  DECLARE_METHOD(GetNodes);
  DECLARE_METHOD(LookupNodes);
  DECLARE_METHOD(Sampling);
  DECLARE_METHOD(Aggregating);
  DECLARE_METHOD(SubGraph);
  DECLARE_METHOD(RandomWalk);
  DECLARE_METHOD(GetCount);
  DECLARE_METHOD(GetDegree);
  DECLARE_METHOD(GetStats);

  Status RunOp(const OpRequest* request, OpResponse* response);

  Status RunDag(const DagRequest* request);
  DECLARE_METHOD(GetDagValues);

  Status Stop();
  Status Report(const StateRequestPb* request);

#undef DECLARE_METHOD

  std::vector<int32_t> GetOwnServers();

private:
  explicit Client(ClientImpl* impl, bool own = true);
  friend Client* NewInMemoryClient();
  friend Client* NewRpcClient(int32_t server_id,
                              bool client_own);

private:
  ClientImpl* impl_;
  bool        own_;
};

Client* NewInMemoryClient();
Client* NewRpcClient(int32_t server_id = -1,
                     bool client_own = false);

}  // namespace graphlearn

#endif  // GRAPHLEARN_INCLUDE_CLIENT_H_
