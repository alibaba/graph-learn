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
#include "graphlearn/common/base/macros.h"
#include "graphlearn/include/config.h"
#include "graphlearn/platform/env.h"
#include "graphlearn/service/dist/channel_manager.h"
#include "graphlearn/service/dist/coordinator.h"
#include "graphlearn/service/dist/grpc_service.h"
#include "graphlearn/service/dist/naming_engine.h"
#include "graphlearn/service/executor.h"

namespace graphlearn {

namespace {

std::string GetLocalEndpoint(int32_t port) {
  char host_name[128] = {0};
  int ret = gethostname(host_name, sizeof(host_name));
  if (ret < 0) {
    LOG(FATAL) << "gethostname error: " << ret;
    return "";
  }

  hostent* hptr = gethostbyname(host_name);
  if (hptr == NULL) {
    LOG(FATAL) << "gethostbyname error";
    return "";
  }

  int i = 0;
  while (hptr->h_addr_list[i] != NULL) {
    std::string ip = inet_ntoa(*(struct in_addr*)hptr->h_addr_list[i]);
    if (ip != "127.0.0.1") {
      return ip + ":" + std::to_string(port);
    } else {
      ++i;
    }
  }
  return "";
}

}  // anonymous namespace

DistributeService::DistributeService(int32_t server_id,
                                     int32_t server_count,
                                     Env* env,
                                     Executor* executor,
                                     Coordinator* coord)
    : server_id_(server_id),
      server_count_(server_count),
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

  std::string endpoint = GetLocalEndpoint(port_);
  Status s = engine_->Update(server_id_, endpoint);
  LOG_RETURN_IF_NOT_OK(s)

  s = coord_->Start();
  LOG_RETURN_IF_NOT_OK(s)

  while (!coord_->IsStartup()) {
    sleep(1);
  }
  return s;
}

Status DistributeService::Init() {
  Status s = coord_->SetReady();
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
  server_->Shutdown();
  manager_->Stop();
  engine_->Stop();
  return Status::OK();
}

Coordinator* DistributeService::GetCoordinator() {
  return coord_;
}

void DistributeService::StartAndJoin() {
  builder_.SetMaxSendMessageSize(GLOBAL_FLAG(RpcMessageMaxSize));
  builder_.SetMaxReceiveMessageSize(GLOBAL_FLAG(RpcMessageMaxSize));
  builder_.AddListeningPort("0.0.0.0:0",
                            ::grpc::InsecureServerCredentials(),
                            &port_);
  builder_.RegisterService(impl_);

  server_ = builder_.BuildAndStart();
  server_->Wait();
}

}  // namespace graphlearn
