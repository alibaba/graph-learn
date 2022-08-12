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

#include "service/server_impl.h"

#include <cstdlib>
#include "common/base/log.h"
#include "core/graph/graph_store.h"
#include "include/config.h"
#include "platform/env.h"
#include "service/dist/coordinator.h"
#include "service/dist/service.h"
#include "service/executor.h"
#include "service/local/in_memory_service.h"

namespace graphlearn {

ServerImpl::ServerImpl(int32_t server_id,
                       int32_t server_count,
                       const std::string& server_host,
                       const std::string& tracker)
    : server_id_(server_id),
      server_count_(server_count),
      server_host_(server_host),
      in_memory_service_(nullptr),
      dist_service_(nullptr),
      coordinator_(nullptr) {
  InitGoogleLogging();
  SetGlobalFlagServerId(server_id);
  SetGlobalFlagServerCount(server_count);
  SetGlobalFlagTracker(tracker);
}

ServerImpl::~ServerImpl() {
  delete in_memory_service_;
  delete dist_service_;
  delete coordinator_;

  UninitGoogleLogging();
}

void ServerImpl::RegisterBasicService(Env* env, Executor* executor) {
  if (GLOBAL_FLAG(DeployMode) != kLocal) {
    coordinator_ = GetCoordinator(server_id_, server_count_, env);
  }

  if (!in_memory_service_) {
    in_memory_service_ = new InMemoryService(env, executor, coordinator_);
    in_memory_service_->Start();
    LOG(INFO) << "Start InMemoryService OK.";
  }

  if (GLOBAL_FLAG(DeployMode) != kLocal && !dist_service_) {
    dist_service_ = new DistributeService(
      server_id_, server_count_, server_host_, env, executor, coordinator_);
    Status s = dist_service_->Start();
    if (!s.ok()) {
      USER_LOG("Server start failed and exit now.");
      USER_LOG(s.ToString());
      LOG(FATAL) << "DistributeService start failed: " << s.ToString();
      ::exit(-1);
    }
    LOG(INFO) << "Start DistributeService OK"
              << ", server_id:" << server_id_
              << ", server_count:" << server_count_;
  }
}

void ServerImpl::InitBasicService() {
  if (in_memory_service_) {
    in_memory_service_->Init();
  }
  if (dist_service_) {
    Status s = dist_service_->Init();
    if (!s.ok()) {
      USER_LOG("Server init failed and exit now.");
      USER_LOG(s.ToString());
      LOG(FATAL) << "DistributeService init failed: " << s.ToString();
      ::exit(-1);
    }
  }
}

void ServerImpl::BuildBasicService() {
  if (in_memory_service_) {
    in_memory_service_->Build();
  }
  if (dist_service_) {
    Status s = dist_service_->Build();
    if (!s.ok()) {
      USER_LOG("Server build failed and exit now.");
      USER_LOG(s.ToString());
      LOG(FATAL) << "DistributeService build failed: " << s.ToString();
      ::exit(-1);
    }
  }
}

void ServerImpl::StopBasicService() {
  if (in_memory_service_) {
    in_memory_service_->Stop();
  }
  if (dist_service_) {
    Status s = dist_service_->Stop();
    if (!s.ok()) {
      USER_LOG("Server stop failed and exit now.");
      USER_LOG(s.ToString());
      LOG(FATAL) << "DistributeService stop failed: " << s.ToString();
      ::exit(-1);
    }
  }
}

void ServerImpl::StopDagService() {
  if (in_memory_service_) {
    in_memory_service_->StopSampling();
  }
  if (dist_service_) {
    dist_service_->StopSampling();
  }
}

DefaultServerImpl::DefaultServerImpl(int32_t server_id,
                                     int32_t server_count,
                                     const std::string& server_host,
                                     const std::string& tracker)
    : ServerImpl(server_id, server_count, server_host, tracker),
      env_(nullptr),
      graph_store_(nullptr),
      executor_(nullptr) {
  env_ = Env::Default();
  graph_store_ = new GraphStore(env_);
  executor_ = new Executor(env_, graph_store_);
}

DefaultServerImpl::~DefaultServerImpl() {
  env_->ShutdownItraThreadPool();
  delete graph_store_;
  delete executor_;
}

void DefaultServerImpl::Start() {
  LOG(INFO) << "Server starts with mode:" << GLOBAL_FLAG(DeployMode)
            << ", server_id:" << server_id_
            << ", server_count:" << server_count_;
  RegisterBasicService(env_, executor_);
  LOG(INFO) << "Server started.";
  USER_LOG("Server started.");
}

void DefaultServerImpl::Init(const std::vector<io::EdgeSource>& edges,
                             const std::vector<io::NodeSource>& nodes) {
  Status s = graph_store_->Load(edges, nodes);
  if (!s.ok()) {
    USER_LOG("Server load data failed and exit now.");
    USER_LOG(s.ToString());
    LOG(FATAL) << "Server load data failed: " << s.ToString();
    ::exit(-1);
  }
  InitBasicService();
  LOG(INFO) << "Data initialized.";
  USER_LOG("Data initialized.");

  s = graph_store_->Build(edges, nodes);
  if (!s.ok()) {
    USER_LOG("Server build data failed and exit now.");
    USER_LOG(s.ToString());
    LOG(FATAL) << "Server build data failed: " << s.ToString();
    ::exit(-1);
  }

  BuildBasicService();
  LOG(INFO) << "Data is ready for serving.";
  USER_LOG("Data is ready for serving.");

  s = graph_store_->BuildStatistics();
  if (!s.ok()) {
    USER_LOG("Server build statistics failed and exit now.");
    USER_LOG(s.ToString());
    LOG(FATAL) << "Server build statistics failed: " << s.ToString();
    ::exit(-1);
  }  
}

void DefaultServerImpl::Stop() {
  StopBasicService();
  LOG(INFO) << "Server stopped.";
  USER_LOG("Server stopped.");
}

const Counts& DefaultServerImpl::GetStats() const {
  return graph_store_->GetStatistics().GetCounts();
}

void DefaultServerImpl::StopSampling() {
  StopDagService();
}

ServerImpl* NewDefaultServerImpl(int32_t server_id,
                                 int32_t server_count,
                                 const std::string& server_host,
                                 const std::string& tracker) {
  return new DefaultServerImpl(server_id, server_count, server_host, tracker);
}

#ifndef OPEN_ACTOR_ENGINE
ServerImpl* NewActorServerImpl(int32_t server_id,
                               int32_t server_count,
                               const std::string& server_host,
                               const std::string& tracker) {
  USER_LOG("Hiactor is disabled! Using default server engine.");
  return new DefaultServerImpl(server_id, server_count, server_host, tracker);
}
#endif

}  // namespace graphlearn
