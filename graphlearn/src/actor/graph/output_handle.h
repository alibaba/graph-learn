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

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "actor/graph/graph_actor.act.h"

#include "actor/graph/loader_config.h"
#include "actor/utils.h"
#include "seastar/core/alien.hh"

namespace graphlearn {
namespace actor {

template <typename Value, typename Wrapper>
class OutputHandle {
public:
  OutputHandle(unsigned init_id);

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
  std::vector<std::shared_ptr<GraphActorRef>> refs_;
};

using NodeOutputHandle = OutputHandle<io::NodeValue, UpdateNodesRequestWrapper>;
using EdgeOutputHandle = OutputHandle<io::EdgeValue, UpdateEdgesRequestWrapper>;

template <typename Value, typename Wrapper>
inline
OutputHandle<Value, Wrapper>::OutputHandle(unsigned init_id)
    : batch_size_(GLOBAL_FLAG(DataInitBatchSize)),
      local_shard_num_(brane::local_shard_count()),
      cursor_id_(init_id) {
  refs_.reserve(brane::global_shard_count());
  buffers_.resize(brane::global_shard_count());

  auto fut = seastar::alien::submit_to(0, [this] {
    for (uint32_t i = 0; i < brane::global_shard_count(); ++i) {
      auto builder = brane::scope_builder(i);
      refs_.emplace_back(builder.new_ref<
        GraphActorRef>(LoaderConfig::graph_actor_id));
    }
    return seastar::make_ready_future<>();
  });

  fut.wait();
}

template <typename Value, typename Wrapper>
inline
void OutputHandle<Value, Wrapper>::Push(Value &value,
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
  for (uint32_t i = 0; i < brane::global_shard_count(); ++i) {
    seastar::alien::run_on(i % local_shard_num_, [ref = refs_[i]] {
      ref->ReceiveEOS();
    });
  }
}

template <typename Value, typename Wrapper>
inline void OutputHandle<Value, Wrapper>::
AlienSend(uint32_t id, UpdateNodesRequestWrapper&& request) {
  seastar::alien::run_on(cursor_id_, [this, ref = refs_[id],
      request = std::move(request)] () mutable {
    ref->UpdateNodes(std::move(request));
  });
  cursor_id_ = (cursor_id_ + 1) % local_shard_num_;
}

template <typename Value, typename Wrapper>
inline void OutputHandle<Value, Wrapper>::
AlienSend(uint32_t id, UpdateEdgesRequestWrapper&& request) {
  seastar::alien::run_on(cursor_id_, [this, ref = refs_[id],
      request = std::move(request)] () mutable {
    ref->UpdateEdges(std::move(request));
  });
  cursor_id_ = (cursor_id_ + 1) % local_shard_num_;
}

}  // namespace actor
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_GRAPH_OUTPUT_HANDLE_H_
