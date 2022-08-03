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

#ifndef GRAPHLEARN_ACTOR_UTILS_H_
#define GRAPHLEARN_ACTOR_UTILS_H_

#include <string>
#include <utility>
#include <vector>

#include "hiactor/core/shard-config.hh"

#include "actor/params.h"

namespace graphlearn {
namespace actor {

struct ShardDataInfo {
  int64_t  data_size;
  uint32_t shard_id;

  ShardDataInfo(int64_t data_size, uint32_t shard_id)
    : data_size(data_size), shard_id(shard_id) {}
};

using ShardDataInfoVecT = std::vector<ShardDataInfo>;

inline
ActorIdType MakeActorGUID(uint32_t base, uint32_t id) {
  // high 16 bits represent base id, low 16 bits represent
  // self id, set the high 16 bits of id to 0
  return base << 16 | (id & 0XFFFF);
}

inline
ShardIdType WhichShard(int64_t id) {
  return id % hiactor::global_shard_count();
}

inline
bool DataSizeLess(const ShardDataInfo& node1,
                  const ShardDataInfo& node2) {
  if (node1.data_size != node2.data_size) {
    return node1.data_size < node2.data_size;
  }
  return node1.shard_id < node2.shard_id;
}

}  // namespace actor
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_UTILS_H_
