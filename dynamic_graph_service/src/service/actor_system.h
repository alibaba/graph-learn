/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef DGS_SERVICE_ACTOR_SYSTEM_H_
#define DGS_SERVICE_ACTOR_SYSTEM_H_

#include <thread>

#include "common/options.h"

namespace dgs {

struct ActorSystem {
public:
  ActorSystem(WorkerType worker_type,
              uint32_t worker_id,
              uint32_t num_workers,
              uint32_t num_local_shards);

  ~ActorSystem();

private:
  static void LaunchWorker(ActorSystem* self);

private:
  const WorkerType worker_type_;
  const uint32_t   worker_id_;
  const uint32_t   num_workers_;
  const uint32_t   num_local_shards_;
  std::unique_ptr<std::thread> main_thread_;
};

}  // namespace dgs

#endif  // DGS_SERVICE_ACTOR_SYSTEM_H_
