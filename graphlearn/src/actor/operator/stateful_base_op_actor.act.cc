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

#include "actor/operator/stateful_base_op_actor.act.h"

#include "actor/graph/sharded_graph_store.h"
#include "core/operator/op_registry.h"

namespace graphlearn {
namespace actor {

StatefulBaseOperatorActor::StatefulBaseOperatorActor(
    brane::actor_base *exec_ctx,
    const brane::byte_t *addr,
    const void* arg)
    : brane::stateful_actor(exec_ctx, addr) {
  const char* op_name = reinterpret_cast<const char*>(arg);
  impl_ = (*op::OpRegistry::OpRegistry::GetInstance()->Lookup(op_name))();
  impl_->Set(ShardedGraphStore::Get().OnShard(brane::local_shard_id()));
}

StatefulBaseOperatorActor::~StatefulBaseOperatorActor() {
}

}  // namespace actor
}  // namespace graphlearn

