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

#include "actor/graph/loader_config.h"
#include "actor/graph/loader_status.h"

#include "actor/generated/graph/sync_actor_ref.act.autogen.h"

namespace graphlearn {
namespace act {

ControlActor::ControlActor(hiactor::actor_base* exec_ctx,
                           const hiactor::byte_t* addr)
    : hiactor::actor(exec_ctx, addr, false /* stateful */),
      received_eos_number_(0) {
}

ControlActor::~ControlActor() = default;

void ControlActor::ReceiveEOS() {
  if (++received_eos_number_ == hiactor::local_shard_count()) {
    // global shard 0 is the location of SyncActor.
    auto builder = hiactor::scope_builder(0);
    auto actor_ref = builder.build_ref<SyncActor_ref>(
        LoaderConfig::sync_actor_id);
    actor_ref.ReceiveEOS(hiactor::Integer(hiactor::global_shard_id()));
  }
}

void ControlActor::StopActor() {
  DataLoaderStatus::Get()->NotifyFinished();
  // TODO(@goldenleaves): kill other loader actors.
}

}  // namespace act
}  // namespace graphlearn
