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

#ifndef GRAPHLEARN_ACTOR_GRAPH_OUTPUT_HANDLE_H_
#define GRAPHLEARN_ACTOR_GRAPH_OUTPUT_HANDLE_H_

#include <functional>
#include <memory>
#include <vector>

#include "seastar/core/alien.hh"

#include "actor/graph/loader_config.h"
#include "actor/service/actor_alien.h"
#include "actor/utils.h"

#include "actor/generated/graph/graph_actor_ref.act.autogen.h"

namespace graphlearn {
namespace act {

template <typename Value, typename Wrapper>
class OutputHandle {
public:
  explicit OutputHandle(unsigned init_id);

  void Push(Value &value, const io::SideInfo& side_info);
  void FlushAll();
  void NotifyFinished();

private:
  void AlienSend(uint32_t shard_id, UpdateNodesRequestWrapper&& request);
  void AlienSend(uint32_t shard_id, UpdateEdgesRequestWrapper&& request);

  inline ShardIdType HashPartition(const io::NodeValue& value) {
    return WhichShard(value.id);
  }

  inline ShardIdType HashPartition(const io::EdgeValue& value) {
    return WhichShard(value.src_id);
  }

private:
  std::vector<Wrapper> buffers_;
  const uint32_t       batch_size_;
  unsigned             local_shard_num_;
  unsigned             cursor_id_;
  std::vector<std::shared_ptr<GraphActor_ref>> refs_;
};

using NodeOutputHandle = OutputHandle<io::NodeValue, UpdateNodesRequestWrapper>;
using EdgeOutputHandle = OutputHandle<io::EdgeValue, UpdateEdgesRequestWrapper>;

template <typename Value, typename Wrapper>
inline
OutputHandle<Value, Wrapper>::OutputHandle(unsigned init_id)
    : batch_size_(GLOBAL_FLAG(DataInitBatchSize)),
      local_shard_num_(hiactor::local_shard_count()),
      cursor_id_(init_id) {
  buffers_.resize(hiactor::global_shard_count());
  refs_.reserve(hiactor::global_shard_count());
  for (uint32_t i = 0; i < hiactor::global_shard_count(); ++i) {
    auto builder = hiactor::scope_builder(i);
    refs_.emplace_back(builder.new_ref<
      GraphActor_ref>(LoaderConfig::graph_actor_id));
  }
}

template <typename Value, typename Wrapper>
inline
void OutputHandle<Value, Wrapper>::Push(
    Value &value,
    const io::SideInfo& side_info) {
  int32_t shard_id = HashPartition(value);
  auto &request = buffers_[shard_id];
  if (request.Empty()) {
    request.Set(side_info);
  }

  request.Append(&value);
  if (request.Size() >= batch_size_) {
    AlienSend(shard_id, std::move(request));
  }
}

template <typename Value, typename Wrapper>
inline
void OutputHandle<Value, Wrapper>::FlushAll() {
  for (size_t i = 0; i < buffers_.size(); ++i) {
    auto &request = buffers_[i];
    if (!request.Empty()) {
      AlienSend(i, std::move(request));
    }
  }
}

template <typename Value, typename Wrapper>
inline
void OutputHandle<Value, Wrapper>::NotifyFinished() {
  FlushAll();
  for (uint32_t i = 0; i < hiactor::global_shard_count(); ++i) {
    seastar::alien::run_on(
        *default_alien,
        i % local_shard_num_,
        [this, ref = refs_[i]] {
      ref->ReceiveEOS();
    });
  }
}

template <typename Value, typename Wrapper>
inline void OutputHandle<Value, Wrapper>::
AlienSend(uint32_t id, UpdateNodesRequestWrapper&& request) {
  seastar::alien::run_on(
      *default_alien,
      cursor_id_,
      [this, ref = refs_[id], request = std::move(request)] () mutable {
    ref->UpdateNodes(std::move(request));
  });
  cursor_id_ = (cursor_id_ + 1) % local_shard_num_;
}

template <typename Value, typename Wrapper>
inline void OutputHandle<Value, Wrapper>::
AlienSend(uint32_t id, UpdateEdgesRequestWrapper&& request) {
  seastar::alien::run_on(
      *default_alien,
      cursor_id_,
      [this, ref = refs_[id], request = std::move(request)] () mutable {
    ref->UpdateEdges(std::move(request));
  });
  cursor_id_ = (cursor_id_ + 1) % local_shard_num_;
}

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_GRAPH_OUTPUT_HANDLE_H_
