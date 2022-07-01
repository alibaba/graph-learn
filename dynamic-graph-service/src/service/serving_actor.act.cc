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

#include "service/serving_actor.act.h"

#include "core/execution/dag.h"
#include "core/execution/query_executor.h"

namespace dgs {

ServingActor::ServingActor(hiactor::actor_base* exec_ctx,
                           const hiactor::byte_t* addr)
  : hiactor::stateful_actor(exec_ctx, addr),
    sample_store_(nullptr) {
}

seastar::future<actor::Void> ServingActor::ExecuteAdminOperation(
    AdminRequest&& req) {
  switch (req.operation) {
    case AdminOperation::PAUSE: {
      // TODO(xiaoming):
      break;
    }
    case AdminOperation::RESUME: {
      // TODO(xiaoming):
      break;
    }
    case AdminOperation::INIT: {
      InitializeImpl(req);
      break;
    }
  }
  return seastar::make_ready_future<actor::Void>();
}

void ServingActor::InitializeImpl(const AdminRequest& req) {
  LOG(INFO) << "Install Query on global shard " << actor::GlobalShardId();
  auto *param = dynamic_cast<ServingInitPayload*>(req.payload.get());
  // Arm sample store.
  sample_store_ = param->sample_store();
  // install Dag locally.
  dag_ = std::make_unique<execution::Dag>(param->query_plan());
}

seastar::future<QueryResponse> ServingActor::RunQuery(
    RunQueryRequest&& req) {  // NOLINT
  dag_->Reset();
  // FIXME(@xmqin):
  return seastar::do_with(execution::QueryExecutor(dag_.get()),
      [this, vid = req.vid()] (auto& executor) {
    return executor.Execute(vid, sample_store_);
  });
}

}  // namespace dgs
