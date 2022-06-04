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

#include "graphlearn/service/dist/channel_manager.h"

#include <unistd.h>
#include <memory>
#include <unordered_map>

#include "graphlearn/common/base/log.h"
#include "graphlearn/common/string/string_tool.h"
#include "graphlearn/common/threading/sync/lock.h"
#include "graphlearn/include/config.h"
#include "graphlearn/platform/env.h"
#include "graphlearn/service/dist/load_balancer.h"
#include "graphlearn/service/dist/naming_engine.h"

namespace graphlearn {

ChannelManager* ChannelManager::GetInstance() {
#if defined(WITH_VINEYARD)
  static std::unordered_map<uint64_t, std::shared_ptr<ChannelManager>> managers;
  uint64_t key = GLOBAL_FLAG(VineyardGraphID);
  if (managers.find(key) == managers.end()) {
    managers[key] = std::shared_ptr<ChannelManager>(new ChannelManager());
  }
  return managers[key].get();
#else
  static ChannelManager manager;
  return &manager;
#endif
}

ChannelManager::ChannelManager() {
  stopped_.store(false);
  channels_.resize(GLOBAL_FLAG(ServerCount), nullptr);

  engine_ = NamingEngine::GetInstance();
  if (GLOBAL_FLAG(TrackerMode) == kRpc) {
    LiteString s(GLOBAL_FLAG(ServerHosts));
    engine_->Update(strings::Split(s, ","));
  }
  balancer_ = NewRoundRobinBalancer(GLOBAL_FLAG(ServerCount));

  auto tp = Env::Default()->ReservedThreadPool();
  tp->AddTask(NewClosure(this, &ChannelManager::Refresh));
}

ChannelManager::~ChannelManager() {
  if (!stopped_) {
    Stop();
  }

  for (size_t i = 0; i < channels_.size(); ++i) {
    delete channels_[i];
  }

  delete balancer_;
}

void ChannelManager::SetCapacity(int32_t capacity) {
  ScopedLocker<std::mutex> _(&mtx_);
  if (!channels_.empty()) {
    channels_.resize(capacity, nullptr);
  }
}

void ChannelManager::Stop() {
  ScopedLocker<std::mutex> _(&mtx_);
  bool to_stop = true;
  for (size_t i = 0; i < channels_.size(); ++i) {
    if (!channels_[i]->IsStopped()) {
      to_stop = false;
    }
  }
  if (to_stop) {
    engine_->Stop();
    stopped_.store(true);
    sleep(1);
  }
}

GrpcChannel* ChannelManager::ConnectTo(int32_t server_id) {
  if (server_id >= channels_.size()) {
    LOG(FATAL) << "Server id out of range and aborted: " << server_id;
    return nullptr;
  }

  if (channels_[server_id] == nullptr) {
    ScopedLocker<std::mutex> _(&mtx_);
    if (channels_[server_id] == nullptr) {
      std::string endpoint = GetEndpoint(server_id);
      channels_[server_id] = new GrpcChannel(endpoint);
    }
  }
  return channels_[server_id];
}

GrpcChannel* ChannelManager::AutoSelect() {
  Status s = balancer_->Calc(GLOBAL_FLAG(ClientCount), 1);
  if (!s.ok()) {
    return nullptr;
  }
  std::vector<int32_t> servers;
  s = balancer_->GetPart(GLOBAL_FLAG(ClientId), &servers);
  if (!s.ok() || servers.empty()) {
    return nullptr;
  }

  LOG(INFO) << "Auto select server: " << servers[0];
  return ConnectTo(servers[0]);
}

std::vector<int32_t> ChannelManager::GetOwnServers() {
  std::vector<int32_t> servers;
  balancer_->GetPart(GLOBAL_FLAG(ClientId), &servers);
  return servers;
}

std::string ChannelManager::GetEndpoint(int32_t server_id) {
  if (engine_->Size() < channels_.size()) {
    LOG(WARNING) << "Waiting for all servers started: "
                 << engine_->Size() << "/" << channels_.size();
    return "";
  }

  int32_t retry = 0;
  std::string endpoint = engine_->Get(server_id);
  while (retry < GLOBAL_FLAG(RetryTimes) && endpoint.empty()) {
    sleep(1 << retry);
    endpoint = engine_->Get(server_id);
    ++retry;
  }
  if (endpoint.empty()) {
    LOG(WARNING) << "Not found endpoint for server: " << server_id;
  }
  return endpoint;
}

void ChannelManager::Refresh() {
  while (!stopped_.load()) {
    {
      ScopedLocker<std::mutex> _(&mtx_);
      if (stopped_.load()) {
        break;
      }
      for (size_t i = 0; i < channels_.size(); ++i) {
        if (channels_[i] && channels_[i]->IsBroken()) {
          std::string endpoint = engine_->Get(i);
          if (!endpoint.empty()) {
            LOG(WARNING) << "Reset channel " << i << " with " << endpoint;
            channels_[i]->Reset(endpoint);
          }
        }
      }
    }
    sleep(1);
  }
}

}  // namespace graphlearn
