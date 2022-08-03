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

#include "actor/graph/control_actor.act.h"

#include <utility>
#include "actor/graph/loader_config.h"
#include "actor/graph/loader_status.h"
#include "actor/graph/sync_actor.act.h"
#include "include/config.h"
#include "seastar/core/sleep.hh"

namespace graphlearn {
namespace actor {

ControlActor::ControlActor(brane::actor_base *exec_ctx,
                           const brane::byte_t *addr,
                           const void*)
    : stateful_actor(exec_ctx, addr),
      received_eos_number_(0) {
}

ControlActor::~ControlActor() {
}

void ControlActor::ReceiveEOS() {
  if (++received_eos_number_ == brane::local_shard_count()) {
    // global shard 0 is the location of SyncActor.
    auto builder = brane::scope_builder(0);
    auto actor_ref = builder.build_ref<SyncActorRef>(
      LoaderConfig::sync_actor_id);
    actor_ref.ReceiveEOS(brane::Integer(brane::global_shard_id()));
  }
}

void ControlActor::StopActor() {
  DataLoaderStatus::Get()->NotifyFinished();
  // TODO(xiaoming.qxm): kill other loader actors.
}

}  // namespace actor
}  // namespace graphlearn
