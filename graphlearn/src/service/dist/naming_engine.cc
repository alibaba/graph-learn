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

#include "graphlearn/service/dist/naming_engine.h"

#include <unistd.h>
#include <memory>
#include <sstream>
#include "graphlearn/common/base/log.h"
#include "graphlearn/common/base/macros.h"
#include "graphlearn/common/threading/sync/lock.h"
#include "graphlearn/include/config.h"
#include "graphlearn/include/constants.h"

namespace graphlearn {

NamingEngine* NamingEngine::GetInstance() {
  if (GLOBAL_FLAG(TrackerMode) == kRpc) {
    static SpecNamingEngine engine;
    return &engine;
  } else {
    static FSNamingEngine engine;
    return &engine;
  }
}

NamingEngine::NamingEngine() : size_(0) {
}

void NamingEngine::SetCapacity(int32_t capacity) {
  ScopedLocker<std::mutex> _(&mtx_);
  endpoints_.resize(capacity);
}

int32_t NamingEngine::Size() const {
  return size_;
}

std::string NamingEngine::Get(int32_t server_id) {
  ScopedLocker<std::mutex> _(&mtx_);
  if (server_id < endpoints_.size()) {
    return endpoints_.at(server_id);
  } else {
    return "";
  }
}

Status NamingEngine::Update(const std::vector<std::string>& endpoints) {
  endpoints_ = endpoints;
  size_ = endpoints.size();
  std::stringstream ss;
  for (auto& endpoint : endpoints) {
    ss << ", " << endpoint;
  }
  LOG(INFO) << "Update endpoints:" << ss.str();
  return Status::OK();
}

Status NamingEngine::Update(int32_t server_id,
                            const std::string& endpoint) {
  if (server_id < endpoints_.size()) {
    endpoints_[server_id] = endpoint;
    LOG(INFO) << "Update endpoint: " << endpoint
              << " for server: " << server_id;
  }
  return Status::OK();
}

}  // namespace graphlearn
