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

#ifndef GRAPHLEARN_ACTOR_SERVICE_ACTOR_SERVICE_H_
#define GRAPHLEARN_ACTOR_SERVICE_ACTOR_SERVICE_H_

#include <thread>  // NOLINT [build/c++11]
#include "service/dist/naming_engine.h"
#include "service/dist/coordinator.h"
#include "include/status.h"

namespace graphlearn {
namespace actor {

class ActorService {
public:
  ActorService(int32_t server_id,
               int32_t server_count,
               Coordinator* coord);
  ~ActorService();

  Status Start();
  Status Init();
  Status Build();
  Status Stop();

private:
  Status StartActorSystem(bool distributed_mode);
  void WriteServerConfigToTmpFile(std::unique_ptr<NamingEngine>&& engine);

private:
  std::thread  actor_system_;
  int32_t      server_id_;
  int32_t      server_count_;
  Coordinator* coord_;
};

}  // namespace actor
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_SERVICE_ACTOR_SERVICE_H_
