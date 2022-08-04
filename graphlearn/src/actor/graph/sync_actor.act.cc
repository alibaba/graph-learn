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

#include "actor/graph/sync_actor.act.h"

#include <utility>
#include "actor/graph/loader_config.h"
#include "actor/graph/control_actor.act.h"
#include "include/config.h"
#include "seastar/core/sleep.hh"

namespace graphlearn {
namespace act {

SyncActor::SyncActor(brane::actor_base *exec_ctx,
                     const brane::byte_t *addr,
                     const void*)
    : stateful_actor(exec_ctx, addr),
      received_eos_number_(0) {
  control_actor_id_.reserve(GLOBAL_FLAG(ServerCount));
}

SyncActor::~SyncActor() {
}

void SyncActor::ReceiveEOS(brane::Integer&& source_id) {
  control_actor_id_.push_back(source_id.val);
  if (++received_eos_number_ == GLOBAL_FLAG(ServerCount)) {
    for (size_t i = 0; i < control_actor_id_.size(); ++i) {
      auto builder = brane::scope_builder(control_actor_id_[i]);
      auto actor_ref = builder.build_ref<ControlActorRef>(
        LoaderConfig::control_actor_id);
      actor_ref.StopActor();
    }
  }
}

}  // namespace act
}  // namespace graphlearn
