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

#include <unistd.h>
#include <cstdlib>
#include <memory>
#include <vector>
#include "actor/graph/sharded_graph_store.h"
#include "actor/service/actor_service.h"
#include "common/base/log.h"
#include "include/config.h"
#include "platform/env.h"
#include "service/dist/coordinator.h"
#include "service/dist/service.h"
#include "service/executor.h"
#include "service/local/in_memory_service.h"

namespace graphlearn {
namespace act {

class ActorService;

class ActorServerImpl : public ServerImpl {
public:
  ActorServerImpl(int32_t server_id,
                  int32_t server_count,
                  const std::string& server_host,
                  const std::string& tracker);
  ~ActorServerImpl() override;

  void Start() override;
  void Init(const std::vector<io::EdgeSource>& edges,
            const std::vector<io::NodeSource>& nodes) override;
  void Stop() override;

private:
  void RegisterActorService();
  void InitActorService();
  void BuildActorService();
  void StopActorService();

private:
  Env*                 env_;
  Executor*            executor_;
  actor::ActorService* actor_service_;
};

ActorServerImpl::ActorServerImpl(int32_t server_id,
                                 int32_t server_count,
                                 const std::string& server_host,
                                 const std::string& tracker)
    : ServerImpl(server_id, server_count, server_host, tracker),
      env_(nullptr),
      executor_(nullptr),
      actor_service_(nullptr) {
  env_ = Env::Default();
  actor::ShardedGraphStore::Get().Init(env_);
  executor_ = new Executor(env_, nullptr);
}

ActorServerImpl::~ActorServerImpl() {
  env_->ShutdownItraThreadPool();
  delete executor_;
  delete actor_service_;
}

void ActorServerImpl::Start() {
  LOG(INFO) << "Server starts with mode:" << GLOBAL_FLAG(DeployMode)
            << ", server_id:" << server_id_
            << ", server_count:" << server_count_;
  RegisterBasicService(env_, executor_);
  RegisterActorService();
  LOG(INFO) << "Server started.";
  USER_LOG("Server started.");
}

void ActorServerImpl::Init(const std::vector<io::EdgeSource>& edges,
                           const std::vector<io::NodeSource>& nodes) {
  Status s = actor::ShardedGraphStore::Get().Load(edges, nodes);
  if (!s.ok()) {
    USER_LOG("Server load data failed and exit now.");
    USER_LOG(s.ToString());
    LOG(FATAL) << "Server load data failed: " << s.ToString();
    ::exit(-1);
  }
  InitBasicService();
  InitActorService();
  LOG(INFO) << "Data initialized.";
  USER_LOG("Data initialized.");

  s = actor::ShardedGraphStore::Get().Build();
  if (!s.ok()) {
    USER_LOG("Server build data failed and exit now.");
    USER_LOG(s.ToString());
    LOG(FATAL) << "Server build data failed: " << s.ToString();
    ::exit(-1);
  }
  BuildBasicService();
  BuildActorService();
  LOG(INFO) << "Data is ready for serving.";
  USER_LOG("Data is ready for serving.");
}

void ActorServerImpl::Stop() {
  StopActorService();
  StopBasicService();
  LOG(INFO) << "Server stopped.";
  USER_LOG("Server stopped.");
}

void ActorServerImpl::RegisterActorService() {
  if (actor_service_) {
    return;
  }

  actor_service_ = new actor::ActorService(
    server_id_, server_count_, coordinator_);
  Status s = actor_service_->Start();
  if (!s.ok()) {
    USER_LOG("Server start failed and exit now.");
    USER_LOG(s.ToString());
    LOG(FATAL) << "ActorService start failed: " << s.ToString();
    ::exit(-1);
  }
  LOG(INFO) << "Start ActorService OK"
            << ", server_id:" << server_id_
            << ", server_count:" << server_count_;
}

void ActorServerImpl::InitActorService() {
  if (!actor_service_) {
    return;
  }

  Status s = actor_service_->Init();
  if (!s.ok()) {
    USER_LOG("Server init failed and exit now.");
    USER_LOG(s.ToString());
    LOG(FATAL) << "ActorService init failed: " << s.ToString();
    ::exit(-1);
  }
}

void ActorServerImpl::BuildActorService() {
  if (!actor_service_) {
    return;
  }

  Status s = actor_service_->Build();
  if (!s.ok()) {
    USER_LOG("Server build failed and exit now.");
    USER_LOG(s.ToString());
    LOG(FATAL) << "ActorService build failed: " << s.ToString();
    ::exit(-1);
  }
}

void ActorServerImpl::StopActorService() {
  if (!actor_service_) {
    return;
  }

  Status s = actor_service_->Stop();
  if (!s.ok()) {
    USER_LOG("ActorService stop failed and exit now.");
    USER_LOG(s.ToString());
    LOG(FATAL) << "ActorService stop failed: " << s.ToString();
    ::exit(-1);
  }
}

}  // namespace act

ServerImpl* NewActorServerImpl(int32_t server_id,
                               int32_t server_count,
                               const std::string& server_host,
                               const std::string& tracker) {
  return new actor::ActorServerImpl(
    server_id, server_count, server_host, tracker);
}

}  // namespace graphlearn