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

#ifndef DGS_COMMON_PARTITION_ROUTER_H_
#define DGS_COMMON_PARTITION_ROUTER_H_

#include <cassert>
#include <vector>
#include <unordered_map>

#include "common/typedefs.h"

namespace dgs {

struct RoutingUpdate {
  PartitionId pid;
  ShardId     sid;
  RoutingUpdate() = default;
  RoutingUpdate(PartitionId pid, ShardId sid) : pid(pid), sid(sid) {}
};

class PartitionRouter {
public:
  PartitionRouter() = default;
  explicit PartitionRouter(const std::vector<ShardId>& pid_to_gsid);
  explicit PartitionRouter(std::vector<ShardId>&& pid_to_gsid);

  PartitionRouter(const PartitionRouter&) = default;
  PartitionRouter(PartitionRouter&&) noexcept = default;
  PartitionRouter& operator=(const PartitionRouter&) = default;
  PartitionRouter& operator=(PartitionRouter&&) noexcept = default;

  ~PartitionRouter() = default;

  /// Get the gsid of a partition
  ShardId GetGlobalShardId(PartitionId pid);

  /// Return a copy of the routing information
  std::vector<ShardId> GetRoutingInfo() {
    return pid_to_gsid_;
  }

  uint32_t Size() {
    return pid_to_gsid_.size();
  }
  /// Update the pid-to-gsid mapping
  void UpdatePartitionRoutingInfo(const std::vector<RoutingUpdate>& updates);
  void UpdatePartitionRoutingInfo(const RoutingUpdate& update);

private:
  std::vector<ShardId> pid_to_gsid_;
};

}  // namespace dgs

#endif  // DGS_COMMON_PARTITION_ROUTER_H_
