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

#include "actor/runner/tape_dispatcher.h"

#include <algorithm>
#include "actor/params.h"
#include "actor/graph/sharded_graph_store.h"
#include "actor/utils.h"
#include "actor/tensor_map.h"
#include "common/base/log.h"
#include "seastar/core/alien.hh"

namespace graphlearn {
namespace act {

class DagActorRefManager {
public:
  DagActorRefManager(const std::vector<ActorIdType> *dag_actor_ids,
                     const ShardIdType global_shard_id)
      : size_(dag_actor_ids->size()), cursor_(0) {
    auto fut = seastar::alien::submit_to(
          global_shard_id % brane::local_shard_count(),
          [this, dag_actor_ids, global_shard_id] {
      brane::scope_builder builder{global_shard_id};
      for (auto id : *dag_actor_ids) {
        refs_.push_back(builder.new_ref<DagActorRef>(id));
      }
      return seastar::make_ready_future<>();
    });
    fut.wait();
  }

  ~DagActorRefManager() {
    for (auto ref : refs_) {
      delete ref;
    }
  }

  DagActorRef* GetRef() {
    auto ref = refs_[cursor_];
    cursor_ = (cursor_ + 1) % size_;
    return ref;
  }

private:
  std::vector<DagActorRef*> refs_;
  uint32_t cursor_;
  const uint32_t size_;
};

class RoundRobinTapeDispatcher : public TapeDispatcher {
public:
  RoundRobinTapeDispatcher(const std::vector<ActorIdType> *dag_actor_ids)
    : TapeDispatcher(dag_actor_ids), cur_shard_id_(0) {}
  
  ~RoundRobinTapeDispatcher() override = default;

  void Dispatch(Tape *tape) override {
    auto runner_ref = dag_runner_refs_[cur_shard_id_]->GetRef();
    seastar::alien::run_on(cur_shard_id_, [runner_ref, tape] {
      runner_ref->RunOnce(TapeHolder(tape));
    });

    cur_shard_id_ = (cur_shard_id_ + 1) % local_shards_;
  }

private:
  uint32_t cur_shard_id_;
};

class OrderedTapeDispatcher : public TapeDispatcher {
public:
  OrderedTapeDispatcher(const std::vector<ActorIdType> *dag_actor_ids,
                        const DagNode* root);
  ~OrderedTapeDispatcher() override = default;
  void Dispatch(Tape *tape) override;

private:
  void MapBatchIdToShardId(const DagNode* root);
  int64_t GetShardDataInfo(const DagNode* root, ShardDataInfoVecT* info_vec);

private:
  uint64_t              total_batches_;
  uint32_t              cur_idx_;
  // the idx of this vector is the batch_id
  // the value of this idx is the shard id
  // e.g [2,3,0,1]; batch 0 should send to shard 2
  std::vector<uint32_t> batch_to_shard_;
};

TapeDispatcher::TapeDispatcher(const std::vector<ActorIdType> *dag_actor_ids)
    : local_shards_(brane::local_shard_count()) {
  BuildRefs(dag_actor_ids);
}

TapeDispatcher::~TapeDispatcher() {
  for (auto *ptr : dag_runner_refs_) {
    delete ptr;
  }
}

void TapeDispatcher::BuildRefs(const std::vector<ActorIdType> *dag_actor_ids) {
  dag_runner_refs_.reserve(local_shards_);
  for (uint32_t i = 0; i < local_shards_; ++i) {
    ShardIdType global_shard_id = i + brane::machine_info::sid_anchor();
    dag_runner_refs_.push_back(
      new DagActorRefManager(dag_actor_ids, global_shard_id));
  }
}

OrderedTapeDispatcher::OrderedTapeDispatcher(
      const std::vector<ActorIdType> *dag_actor_ids,
      const DagNode* root)
    : TapeDispatcher(dag_actor_ids),
      total_batches_(0), cur_idx_(0) {
  MapBatchIdToShardId(root);
}

void OrderedTapeDispatcher::Dispatch(Tape* tape) {
  if (cur_idx_ < total_batches_) {
    uint32_t local_shard_id = batch_to_shard_[cur_idx_++];
    auto runner_ref = dag_runner_refs_[local_shard_id]->GetRef();
    seastar::alien::run_on(local_shard_id, [runner_ref, tape] {
      runner_ref->RunOnce(TapeHolder(tape));
    });
  } else {
    // this is the last batch (fake tape)
    tape->Fake();
    cur_idx_ = 0;
  }
}

void OrderedTapeDispatcher::MapBatchIdToShardId(const DagNode* root) {
  int32_t batch_size = root->Params().at(kBatchSize).GetInt32(0);
  ShardDataInfoVecT info_vec;
  int64_t total_size = GetShardDataInfo(root, &info_vec);
  std::sort(info_vec.begin(), info_vec.end(), DataSizeLess);

  total_batches_ = total_size / batch_size + (total_size % batch_size ? 1 : 0);
  batch_to_shard_.resize(total_batches_);

  uint32_t batch_id = 0;
  uint32_t offset = 0;
  // For better load balance,
  // Assign batch_id in turn cross between sorted shards
  for (uint32_t i = 0; i < local_shards_; i++) {
    auto shard_total_batch_num = info_vec[i].data_size / batch_size;
    for (auto j = offset; j < shard_total_batch_num; j++) {
      for (uint32_t behind_i = i; behind_i < local_shards_; behind_i++) {
        batch_to_shard_[batch_id++] = info_vec[behind_i].shard_id;
      }
    }
    offset = shard_total_batch_num;
  }

  // Process last few batches
  for (uint32_t i = 0; batch_id < total_batches_; batch_id++) {
    batch_to_shard_[batch_id] = info_vec[i++].shard_id;
  }
}

int64_t OrderedTapeDispatcher::GetShardDataInfo(
    const DagNode* root,
    ShardDataInfoVecT* info_vec) {
  info_vec->reserve(local_shards_);
  int64_t total_size = 0;

  if (root->OpName() == "GetNodes") {
    std::string type = root->Params().at(kNodeType).GetString(0);
    for (uint32_t i = 0; i < local_shards_; ++i) {
      auto noder = ShardedGraphStore::Get().OnShard(i)->GetNoder(type);
      auto total_size_on_shard = noder->GetLocalStorage()->Size();
      total_size += total_size_on_shard;
      info_vec->emplace_back(total_size_on_shard, i);
    }
  } else if (root->OpName() == "GetEdges") {
    // FIXME: verify getting edge data
    std::string type = root->Params().at(kEdgeType).GetString(0);
    for (uint32_t i = 0; i < local_shards_; ++i) {
      auto edger = ShardedGraphStore::Get().OnShard(i)->GetGraph(type);
      auto total_size_on_shard = edger->GetLocalStorage()->GetEdgeCount();
      total_size += total_size_on_shard;
      info_vec->emplace_back(total_size_on_shard, i);
    }
  } else {
    LOG(ERROR) << "Not supported OpName: " << root->OpName();
  }
  return total_size;
}

std::unique_ptr<TapeDispatcher> NewTapeDispatcher(
    const std::vector<ActorIdType> *dag_actor_ids,
    const DagNode* root) {
  std::unique_ptr<TapeDispatcher> dispatcher{nullptr};
  if (root->Params().at(kStrategy).GetString(0) == "random") {
    dispatcher.reset(new RoundRobinTapeDispatcher(dag_actor_ids));
  } else {
    dispatcher.reset(new OrderedTapeDispatcher(dag_actor_ids, root));
  }
  return dispatcher;
}

}  // namespace act
}  // namespace graphlearn