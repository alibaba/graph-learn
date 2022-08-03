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

#include "actor/graph/graph_actor.act.h"

#include <utility>
#include "actor/graph/loader_config.h"
#include "actor/graph/sharded_graph_store.h"
#include "core/graph/graph_store.h"
#include "include/config.h"

namespace graphlearn {
namespace actor {

GraphActor::GraphActor(brane::actor_base *exec_ctx,
                       const brane::byte_t *addr,
                       const void*)
    : stateful_actor(exec_ctx, addr), received_eos_number_(0) {
  data_parser_num_ = GLOBAL_FLAG(ServerCount) * 2 *
    std::max(GLOBAL_FLAG(InterThreadNum), 1);
  store_ = ShardedGraphStore::Get().OnShard(brane::local_shard_id());
}

GraphActor::~GraphActor() {
}

void GraphActor::UpdateNodes(UpdateNodesRequestWrapper&& request) {
  nodes_num_ += request.Size();
  auto noder = store_->GetNoder(request.Type());
  UpdateNodesResponse res;
  noder->UpdateNodes(request.Get(), &res);
}

void GraphActor::UpdateEdges(UpdateEdgesRequestWrapper&& request) {
  edges_num_ += request.Size();
  auto graph = store_->GetGraph(request.Type());
  UpdateEdgesResponse res;
  graph->UpdateEdges(request.Get(), &res);
}

void GraphActor::ReceiveEOS() {
  if (++received_eos_number_ == data_parser_num_) {
    auto builder = brane::scope_builder(brane::machine_info::sid_anchor());
    auto ctrl_ref = builder.build_ref<ControlActorRef>(
      LoaderConfig::control_actor_id);
    ctrl_ref.ReceiveEOS();
  }
}

}  // namespace actor
}  // namespace graphlearn
