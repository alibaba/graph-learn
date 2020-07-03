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

#include "graphlearn/service/dist/coordinator.h"

#include <unistd.h>
#include <memory>
#include <vector>
#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/base/log.h"
#include "graphlearn/common/string/string_tool.h"
#include "graphlearn/include/config.h"
#include "graphlearn/platform/env.h"

namespace graphlearn {

Coordinator::Coordinator(int32_t server_id, int32_t server_count,
                         Env* env)
    : client_count_(-1),
      server_id_(server_id),
      server_count_(server_count),
      state_(kBlank) {
}

Coordinator::~Coordinator() {
  state_ = kStopped;
}

bool Coordinator::IsMaster() const {
  return server_id_ == 0;
}

bool Coordinator::IsStartup() const {
  return (state_ >= kStarted);
}

bool Coordinator::IsInited() const {
  return (state_ >= kInited);
}

bool Coordinator::IsReady() const {
  return (state_ >= kReady);
}

bool Coordinator::IsStopped() const {
  return (state_ >= kStopped);
}


void Coordinator::Refresh() {
  while (state_ < kStopped) {
    if (state_ < kStarted) {
      CheckStarted();
    }
    // if (state_ < kInited) {
    //   CheckInited();
    // }
    if (state_ < kReady) {
      CheckReady();
    }
    if (state_ < kStopped) {
      CheckStopped();
    }
    sleep(1);
  }
}

class CoordinatorCreator {
public:
  CoordinatorCreator() {}
  ~CoordinatorCreator() = default;

  Coordinator* operator() (int32_t server_id, int32_t server_count,
                           Env* env) {
    Coordinator* coord = nullptr;
    if (GLOBAL_FLAG(TrackerMode) < 1) {
      coord = new RPCCoordinator(server_id, server_count, env);
    } else {
      coord = new FSCoordinator(server_id, server_count, env);
    }
    return coord;
  }
};

Coordinator* GetCoordinator(int32_t server_id, int32_t server_count,
                            Env* env){
  static CoordinatorCreator creator;
  return creator(server_id, server_count, env);
}

}  // namespace graphlearn
