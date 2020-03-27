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

#include "graphlearn/service/server_impl.h"

#include <unistd.h>
#include <cstdlib>
#include <memory>
#include <vector>
#include "graphlearn/common/base/log.h"
#include "graphlearn/core/graph/graph_store.h"
#include "graphlearn/include/config.h"
#include "graphlearn/platform/env.h"
#include "graphlearn/service/dist/service.h"
#include "graphlearn/service/executor.h"
#include "graphlearn/service/local/in_memory_service.h"

namespace graphlearn {

ServerImpl::ServerImpl(int32_t server_id,
                       int32_t server_count,
                       const std::string& tracker)
    : server_id_(server_id),
      server_count_(server_count),
      executor_(nullptr),
      graph_store_(nullptr),
      in_memory_service_(nullptr),
      dist_service_(nullptr) {
  InitGoogleLogging();
  SetGlobalFlagServerId(server_id);
  SetGlobalFlagServerCount(server_count);
  SetGlobalFlagTracker(tracker);
  env_ = Env::Default();
  graph_store_ = new GraphStore(env_);
  executor_ = new Executor(env_, graph_store_);
}

ServerImpl::~ServerImpl() {
  delete in_memory_service_;
  delete executor_;
  delete graph_store_;

  UninitGoogleLogging();
}

void ServerImpl::Start() {
  if (GLOBAL_FLAG(DeployMode) >= 1) {
    RegisterInMemoryService();
    RegisterDistributeService();
  } else {
    RegisterInMemoryService();
  }
  USER_LOG("Server started.");
}

void ServerImpl::Init(const std::vector<io::EdgeSource>& edges,
                      const std::vector<io::NodeSource>& nodes) {
  if (graph_store_) {
    graph_store_->Load(edges, nodes);
    graph_store_->Build();
  }
  if (in_memory_service_) {
    in_memory_service_->Init();
  }
  if (dist_service_) {
    Status s = dist_service_->Init();
    if (!s.ok()) {
      LOG(FATAL) << "Service init failed: " << s.ToString();
      ::exit(-1);
    }
  }
  USER_LOG("Data initialized.");
}

void ServerImpl::Stop() {
  if (in_memory_service_) {
    in_memory_service_->Stop();
  }
  if (dist_service_) {
    Status s = dist_service_->Stop();
    if (!s.ok()) {
      LOG(FATAL) << "Service stop failed: " << s.ToString();
      ::exit(-1);
    }
  }
  USER_LOG("Server stopped.");
}

void ServerImpl::RegisterInMemoryService() {
  if (in_memory_service_ == nullptr) {
    in_memory_service_ = new InMemoryService(env_, executor_);
    in_memory_service_->Start();
  }
}

void ServerImpl::RegisterDistributeService() {
  if (dist_service_ == nullptr) {
    dist_service_ = new DistributeService(
      server_id_, server_count_, env_, executor_);
    Status s = dist_service_->Start();
    if (!s.ok()) {
      LOG(FATAL) << "Service start failed: " << s.ToString();
      ::exit(-1);
    }
  }
}

}  // namespace graphlearn
