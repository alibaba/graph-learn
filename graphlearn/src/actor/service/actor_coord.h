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

#ifndef GRAPHLEARN_ACTOR_SERVICE_ACTOR_COORD_H_
#define GRAPHLEARN_ACTOR_SERVICE_ACTOR_COORD_H_

#include <string>
#include "brane/core/coordinator.hh"

namespace graphlearn {

class Coordinator;

namespace act {

class ActorCoordImpl : public brane::coordinator::impl {
public:
  ActorCoordImpl(Coordinator* coord, int32_t server_id, int32_t server_count)
      : brane::coordinator::impl(), coord_(coord),
        server_id_(server_id), server_count_(server_count) {}

  ~ActorCoordImpl() override = default;

  int global_barrier(const std::string& barrier_guid) final;

private:
  Coordinator* coord_;
  int32_t      server_id_;
  int32_t      server_count_;
};

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_SERVICE_ACTOR_COORD_H_
