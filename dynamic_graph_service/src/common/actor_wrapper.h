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

#ifndef DGS_COMMON_ACTOR_WRAPPER_H_
#define DGS_COMMON_ACTOR_WRAPPER_H_

#include "hiactor/core/column_batch.hh"
#include "hiactor/core/dynamic-queue.hh"
#include "hiactor/core/promise_manager.hh"
#include "hiactor/core/reference_base.hh"
#include "hiactor/core/shard-config.hh"
#include "hiactor/net/serializable_queue.hh"
#include "hiactor/util/data_type.hh"
#include "hiactor/util/machine_info.hh"

#include "seastar/core/temporary_buffer.hh"
#include "seastar/core/circular_buffer.hh"

#include "common/constants.h"
#include "common/typedefs.h"

namespace dgs {
namespace act {

using Void = hiactor::Void;
using Bool = hiactor::Boolean;
using Integer = hiactor::Integer;
using BytesBuffer = seastar::temporary_buffer<char>;
using SerializableQueue = hiactor::serializable_queue;
using IdColumnBatch = hiactor::cb::fixed_column_batch<VertexId>;
using VoidPromiseManager = hiactor::pr_manager<>;

template <typename T>
using AsyncQueue = hiactor::dynamic_queue<T>;

template <typename T>
using CircularBuffer = seastar::circular_buffer<T>;

inline uint32_t GlobalShardCount() {
  return hiactor::global_shard_count();
}

inline uint32_t LocalShardCount() {
  return hiactor::local_shard_count();
}

inline uint32_t GlobalShardId() {
  return hiactor::global_shard_id();
}

inline uint32_t LocalShardId() {
  return hiactor::local_shard_id();
}

inline uint32_t GlobalShardIdAnchor() {
  return hiactor::machine_info::sid_anchor();
}

inline uint32_t GetWorkerShardIdAnchor(WorkerId wid) {
  return hiactor::local_shard_count() * wid;
}

// The higher the value, the higher the priority
static_assert(kDataUpdateActorInstId < kServingActorInstId,
    "The data update actor id should be smaller than serving actor");

}  // namespace act
}  // namespace dgs

#endif  // DGS_COMMON_ACTOR_WRAPPER_H_
