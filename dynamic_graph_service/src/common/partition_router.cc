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

#include "common/partition_router.h"

#include "common/actor_wrapper.h"

namespace dgs {

PartitionRouter::PartitionRouter(const std::vector<ShardId>& pid_to_gsid)
  : pid_to_gsid_(pid_to_gsid) {
}

PartitionRouter::PartitionRouter(std::vector<ShardId>&& pid_to_gsid)
  : pid_to_gsid_(std::move(pid_to_gsid)) {
}

ShardId PartitionRouter::GetGlobalShardId(PartitionId pid) {
  assert(pid < pid_to_gsid_.size());
  return pid_to_gsid_[pid];
}

void PartitionRouter::UpdatePartitionRoutingInfo(
    const std::vector<RoutingUpdate>& updates) {
  for (auto& update : updates) {
    UpdatePartitionRoutingInfo(update);
  }
}

void PartitionRouter::UpdatePartitionRoutingInfo(const RoutingUpdate& update) {
  if (pid_to_gsid_.size() <= update.pid) {
    pid_to_gsid_.resize(update.pid + 1);
  }
  pid_to_gsid_[update.pid] = update.sid;
}

}  // namespace dgs
