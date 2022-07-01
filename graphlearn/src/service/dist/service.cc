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

#include "graphlearn/service/dist/service.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include "graphlearn/common/base/host.h"
#include "graphlearn/common/base/macros.h"
#include "graphlearn/include/config.h"
#include "graphlearn/platform/env.h"
#include "graphlearn/service/dist/channel_manager.h"
#include "graphlearn/service/dist/coordinator.h"
#include "graphlearn/service/dist/grpc_service.h"
#include "graphlearn/service/dist/naming_engine.h"
#include "graphlearn/service/executor.h"

namespace graphlearn {

DistributeService::DistributeService(int32_t server_id,
                                     int32_t server_count,
                                     const std::string& server_host,
                                     Env* env,
                                     Executor* executor,
                                     Coordinator* coord)
    : server_id_(server_id),
      server_count_(server_count),
      server_host_(server_host),
      port_(0),
      coord_(coord),
      impl_(nullptr) {
  engine_ = NamingEngine::GetInstance();
  engine_->SetCapacity(server_count);
  manager_ = ChannelManager::GetInstance();
  impl_ = new GrpcServiceImpl(env, executor, coord_);
}

DistributeService::~DistributeService() {
  delete impl_;
}

Status DistributeService::Start() {
  auto tp = Env::Default()->ReservedThreadPool();
  tp->AddTask(NewClosure(this, &DistributeService::StartAndJoin));

  while (port_ == 0) {
    sleep(1);
  }

  Status s;

  if (GLOBAL_FLAG(TrackerMode) == kFileSystem) {
    std::string endpoint = GetLocalEndpoint(port_);
    s = engine_->Update(server_id_, endpoint);
    LOG_RETURN_IF_NOT_OK(s)
  } else {
    // s = engine_->Update(GLOBAL_FLAG(ServerHosts));
    // LOG_RETURN_IF_NOT_OK(s)
    // Do nothing
  }

  s = coord_->Start();
  LOG_RETURN_IF_NOT_OK(s)

  while (!coord_->IsStartup()) {
    sleep(1);
  }
  return s;
}

Status DistributeService::Init() {
  Status s = coord_->Init();
  RETURN_IF_NOT_OK(s)

  while (!coord_->IsInited()) {
    sleep(1);
  }
  return s;
}

Status DistributeService::Build() {
  Status s = coord_->Prepare();
  RETURN_IF_NOT_OK(s)

  while (!coord_->IsReady()) {
    sleep(1);
  }
  return s;
}

Status DistributeService::Stop() {
  while (!coord_->IsStopped()) {
    LOG(WARNING) << "Waiting other servers to stop";
    sleep(1);
  }
  Env::Default()->SetStopping();
  server_->Shutdown();
  manager_->Stop();
  engine_->Stop();
  coord_->Finallize();
  return Status::OK();
}

void DistributeService::StopSampling() {
  Env::Default()->SetStopping();
}

Coordinator* DistributeService::GetCoordinator() {
  return coord_;
}

void DistributeService::StartAndJoin() {
  builder_.SetMaxSendMessageSize(GLOBAL_FLAG(RpcMessageMaxSize));
  builder_.SetMaxReceiveMessageSize(GLOBAL_FLAG(RpcMessageMaxSize));
  if (GLOBAL_FLAG(TrackerMode == kRpc)) {
    builder_.AddListeningPort(server_host_,
                              ::grpc::InsecureServerCredentials(),
                              &port_);
  } else {
    builder_.AddListeningPort("0.0.0.0:0",
                              ::grpc::InsecureServerCredentials(),
                              &port_);
  }
  builder_.RegisterService(impl_);
  server_ = builder_.BuildAndStart();

  int32_t retry = 1;
  while (!server_ && retry < GLOBAL_FLAG(RetryTimes)) {
    sleep(retry++);
    server_ = builder_.BuildAndStart();
  }
  if (!server_) {
    LOG(FATAL) << "Start server failed, please check the environment. "
               << "Endpoint: " << server_host_;
  }
  server_->Wait();
}

}  // namespace graphlearn
