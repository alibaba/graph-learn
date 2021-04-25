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

#include <unistd.h>
#include <memory>
#include <vector>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/common/base/macros.h"
#include "graphlearn/include/client.h"
#include "graphlearn/include/config.h"
#include "graphlearn/platform/env.h"
#include "graphlearn/proto/request.pb.h"
#include "graphlearn/service/dist/coordinator.h"

namespace graphlearn {

RPCCoordinator::RPCCoordinator(int32_t server_id, int32_t server_count,
                               Env* env)
    : Coordinator(server_id, server_count, env),
      reserved_state_(SystemState::kStopped + 1) {
  auto tp = env->ReservedThreadPool();
  tp->AddTask(NewClosure(this, &RPCCoordinator::Refresh));
}

Status RPCCoordinator::Sync(const std::string& barrier) {
  int32_t state = reserved_state_ + 1;
  Status s;
  if (IsMaster()) {
    s = SetState(state, 0);
  } else {
    s = ReportState(0, state, server_id_);
  }
  LOG_RETURN_IF_NOT_OK(s)

  while (state > reserved_state_) {
    CheckState(state, server_count_);
    sleep(1);
  }
  return Status::OK();
}

Status RPCCoordinator::Start() {
  if (IsMaster()) {
    return SetStarted(0);
  } else {
    return ReportState(0, kStarted, server_id_);
  }
}

Status RPCCoordinator::SetStarted(int32_t server_id) {
  return SetState(kStarted, server_id);
}

Status RPCCoordinator::Init() {
  if (IsMaster()) {
    return SetInited(0);
  } else {
    return ReportState(0, kInited, server_id_);
  }
}

Status RPCCoordinator::SetInited(int32_t server_id) {
  return SetState(kInited, server_id);
}

Status RPCCoordinator::Prepare() {
  if (IsMaster()) {
    return SetReady(0);
  } else {
    return ReportState(0, kReady, server_id_);
  }
}

Status RPCCoordinator::SetReady(int32_t server_id) {
  return SetState(kReady, server_id);
}

Status RPCCoordinator::Stop(int32_t client_id, int32_t client_count) {
  return ReportState(0, kStopped, client_id, client_count);
}

Status RPCCoordinator::SetStopped(int32_t client_id, int32_t client_count) {
  client_count_ = client_count;
  return SetState(kStopped, client_id);
}

void RPCCoordinator::Refresh() {
  Coordinator::Refresh();
}

void RPCCoordinator::CheckStarted() {
  CheckState(kStarted, server_count_);
}

void RPCCoordinator::CheckInited() {
  CheckState(kInited, server_count_);
}

void RPCCoordinator::CheckReady() {
  CheckState(kReady, server_count_);
}

void RPCCoordinator::CheckStopped() {
  CheckState(kStopped, client_count_);
}

Status RPCCoordinator::SetState(SystemState state, int32_t id) {
  ScopedLocker<std::mutex> _(&mtx_);
  if (id == -1) {
    state_ = state;
  } else {
    if (state_map_.find(state) == state_map_.end()) {
      std::set<int32_t> idset{};
      state_map_.insert({state, idset});
    }
    state_map_[state].insert(id);
  }
  return Status::OK();
}

Status RPCCoordinator::SetState(int32_t state, int32_t id) {
  ScopedLocker<std::mutex> _(&mtx_);
  if (id == -1) {
    reserved_state_ = state;
  } else {
    if (state_map_.find(state) == state_map_.end()) {
      std::set<int32_t> idset{};
      state_map_.insert({state, idset});
    }
    state_map_[state].insert(id);
  }
  return Status::OK();
}

void RPCCoordinator::CheckState(SystemState state, int32_t count) {
  ScopedLocker<std::mutex> _(&mtx_);
  if (IsMaster() && state_map_[state].size() == count) {
    state_ = state;
    for (int32_t remote_id = 1; remote_id < server_count_; ++remote_id) {
      ReportState(remote_id, state);
    }
  }
}

void RPCCoordinator::CheckState(int32_t state, int32_t count) {
  ScopedLocker<std::mutex> _(&mtx_);
  if (IsMaster() && state_map_[state].size() == count) {
    reserved_state_ = state;
    for (int32_t remote_id = 1; remote_id < server_count_; ++remote_id) {
      ReportState(remote_id, state);
    }
  }
}

Status RPCCoordinator::ReportState(int32_t target, int32_t state,
                                   int32_t id, int32_t count) {
  std::unique_ptr<Client> client(NewRpcClient(target));
  StateRequestPb req;
  req.set_state(state);
  req.set_id(id);
  req.set_count(count);
  return client->Report(&req);
}

}  // namespace graphlearn
