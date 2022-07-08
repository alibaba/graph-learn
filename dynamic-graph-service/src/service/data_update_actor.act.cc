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

#include "service/data_update_actor.act.h"

#include "common/options.h"
#include "common/log.h"

namespace dgs {

DataUpdateActor::DataUpdateActor(hiactor::actor_base* exec_ctx,
                                 const hiactor::byte_t* addr)
  : hiactor::stateless_actor(exec_ctx, addr),
    sample_store_(nullptr),
    num_graph_updates_(0),
    data_log_period_(Options::GetInstance().GetLoggingOptions().data_log_period) {
}

seastar::future<actor::Void>
DataUpdateActor::Update(io::SampleUpdateBatch&& batch) {
  auto updates = batch.GetSampleUpdates();
  for (auto& update : updates) {
    if (update.value.GetView().Type() == RecordType::VERTEX) {
      sample_store_->PutVertex(update.key,
        {update.value.Data(), update.value.Size()});
    } else {
      sample_store_->PutEdge(update.key,
        {update.value.Data(), update.value.Size()});
    }
  }

  if (num_graph_updates_++ % data_log_period_ == 0) {
    LOG(INFO) << "Apply Graph Update on global shard "
              << actor::GlobalShardId()
              << ": #processed_updates is "
              << num_graph_updates_
              << ", sample update batch size is "
              << updates.size();
  }

  return seastar::make_ready_future<actor::Void>();
}

seastar::future<actor::Void>
DataUpdateActor::ExecuteAdminOperation(AdminRequest&& req) {
  switch (req.operation) {
    case AdminOperation::PAUSE: {
      // TODO(@goldenleaves)
    }
    case AdminOperation::RESUME: {
      // TODO(@goldenleaves)
    }
    case AdminOperation::INIT: {
      LOG(INFO) << "Initialize on global shard " << actor::GlobalShardId();
      auto *param = dynamic_cast<ServingInitPayload*>(req.payload.get());
      // Arm sample store.
      sample_store_ = param->sample_store();
    }
  }
  return seastar::make_ready_future<actor::Void>();
}

}  // namespace dgs
