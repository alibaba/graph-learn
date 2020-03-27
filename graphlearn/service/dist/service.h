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

#ifndef GRAPHLEARN_SERVICE_DIST_SERVICE_H_
#define GRAPHLEARN_SERVICE_DIST_SERVICE_H_

#include <cstdint>
#include <memory>
#include "graphlearn/include/status.h"
#include "grpcpp/grpcpp.h"

namespace graphlearn {

class Env;
class Executor;
class Coordinator;
class NamingEngine;
class ChannelManager;
class GrpcServiceImpl;

class DistributeService {
public:
  DistributeService(int32_t server_id, int32_t server_count,
                    Env* env, Executor* executor);
  ~DistributeService();

  Status Start();
  Status Init();
  Status Stop();

  // just for test
  Coordinator* GetCoordinator();

private:
  void StartAndJoin();

private:
  int32_t         server_id_;
  int32_t         server_count_;
  int32_t         port_;
  Coordinator*    coord_;
  NamingEngine*   engine_;
  ChannelManager* manager_;

  GrpcServiceImpl*                impl_;
  ::grpc::ServerBuilder           builder_;
  std::unique_ptr<::grpc::Server> server_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_SERVICE_DIST_SERVICE_H_

