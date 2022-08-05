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

#include "actor/dag/dag_actor.act.h"

#include "seastar/core/loop.hh"

#include "actor/dag/dag_actor_manager.h"
#include "actor/utils.h"
#include "common/base/log.h"

#include "actor/generated/operator/base_op_ref.act.autogen.h"

namespace graphlearn {
namespace act {

namespace {

ShardsPtr<ShardableTensorMap> PartitionInput(
    const NodeProxy* op,
    ShardableTensorMap* req) {
  auto& params = op->Params();
  auto it = params.find(kPartitionKey);
  if (it != params.end()) {
    const std::string& pKey = it->second.GetString(0);
    req->SetPartitionKey(pKey);
  }
  return req->Partition();
}

std::vector<int32_t> GetShardIds(
    const NodeProxy* op,
    const ShardsPtr<ShardableTensorMap>& reqs) {
  std::vector<int32_t> shard_ids;
  if (op->IsSource()) {
    shard_ids.reserve(1);
    shard_ids.push_back(static_cast<int32_t>(hiactor::global_shard_id()));
  } else {
    shard_ids.reserve(reqs->Size());
    int32_t sid = 0;
    ShardableTensorMap* tmp = nullptr;
    while (reqs->Next(&sid, &tmp)) {
      shard_ids.push_back(sid);
    }
    reqs->ResetNext();
  }
  return shard_ids;
}

}  // anonymous namespace

DagActor::DagActor(hiactor::actor_base* exec_ctx, const hiactor::byte_t* addr)
    : hiactor::actor(exec_ctx, addr, false),
      env_(nullptr), stopping_(false) {
  auto& mgr = DagActorManager::GetInstance();
  const auto* dag_params = reinterpret_cast<const DagActorParams*>(
      mgr.GetActorParams(actor_id()));
  dag_proxy_ = DagProxy(dag_params);
}

DagActor::~DagActor() = default;

seastar::future<hiactor::Void> DagActor::RunOnce(TapeHolder&& holder) {
  auto* tape = holder.tape;
  return seastar::do_until([this] { return IsStopping(); },
                           [this, tape] {
    // Get the dag node list that are ready to run.
    auto next = dag_proxy_.Next();
    // Run all the ready nodes parallel.
    return seastar::parallel_for_each(next.begin(), next.end(),
        [this, tape] (int64_t node_id) {
      NodeProxy* cur_op = &(dag_proxy_.Node(node_id));
      ShardableTensorMap* req = BuildInput(cur_op, tape);
      if (__builtin_expect(req != nullptr, true)) {
        return RunInParallel(cur_op, req, tape);
      } else {
        stopping_ = true;
        LOG(ERROR) << "BuildInput for DagNode " << cur_op->GUID() << " failed.";
        return seastar::make_ready_future<>();
      }
    });
  }).then([this, tape] () mutable {
    tape->SetReady();
    dag_proxy_.Reset();
    return seastar::make_ready_future<hiactor::Void>();
  });
}

// Mapping the upstream output to the downstream input
ShardableTensorMap* DagActor::BuildInput(const NodeProxy* op, Tape* tape) {
  Tensor::Map inputs;
  for (auto &in_edge : op->Upstreams()) {
    NodeProxy& upstream_op = dag_proxy_.Node(in_edge.UpstreamGUID());
    auto& tensors = tape->Retrieval(static_cast<int32_t>(upstream_op.GUID()));
    auto joint = in_edge.Joint();
    if (tensors.find(joint.first) != tensors.end()) {
      inputs.emplace(joint.second, tensors.at(joint.first));
    } else {
      LOG(ERROR) << "Current DagNode " << op->GUID()
                 << ": cannot find " << joint.first
                 << " in the upstream output, upstream DagNode: "
                 << in_edge.UpstreamGUID();
      return nullptr;
    }
  }
  return new ShardableTensorMap(std::move(inputs));
}

seastar::future<>
DagActor::RunInParallel(const NodeProxy* op,
                        ShardableTensorMap* input,
                        Tape* tape) {
  ShardsPtr<ShardableTensorMap> shard_reqs = PartitionInput(op, input);
  std::vector<int32_t> shard_ids = GetShardIds(op, shard_reqs);
  ShardsPtr<JoinableTensorMap> shard_rets(
      new Shards<JoinableTensorMap>(shard_reqs->Capacity()));

  return seastar::parallel_for_each(
      shard_ids.begin(), shard_ids.end(), [&] (int32_t shard_id) {
    auto part = shard_reqs->Get(shard_id);
    if (part) {
      return ProcessInShard(op, shard_id, part).then(
          [shard_id, shard_rets] (JoinableTensorMap *res) {
        shard_rets->Add(shard_id, res, true);
      });
    } else {
      LOG(WARNING) << "Input of this partition is empty: " << shard_id;
      return seastar::make_ready_future<>();
    }
  }).then([op, input, shard_rets, tape, shard_reqs] {
    JoinableTensorMap res;
    shard_rets->StickerPtr()->CopyFrom(*(shard_reqs->StickerPtr()));
    res.Stitch(shard_rets);
    tape->Record(static_cast<int32_t>(op->GUID()), std::move(res.tensors_));
    delete input;
    return seastar::make_ready_future<>();
  });
}

seastar::future<JoinableTensorMap*>
DagActor::ProcessInShard(const NodeProxy* op,
                         int32_t shard_id,
                         ShardableTensorMap* req) {
  auto* ref = op->OnShard(shard_id);
  if (!ref) {
    LOG(ERROR) << "guidToActorRef for shard " << shard_id
                << " of node " << op->GUID() << "is nullptr.";
    stopping_ = true;
    return seastar::make_ready_future<JoinableTensorMap*>(nullptr);
  }

  return ref->Process(TensorMap(std::move(req->tensors_))).then_wrapped(
    [this] (seastar::future<TensorMap> response) {
      JoinableTensorMap *res = nullptr;
      if (__builtin_expect(response.failed(), false)) {
        response.ignore_ready_future();
        stopping_ = true;
      } else {
        // Note that we only can get0 once
        auto out = response.get0();
        res = new JoinableTensorMap(std::move(out.tensors_));
      }
    return seastar::make_ready_future<JoinableTensorMap*>(res);
  });
}

bool DagActor::IsStopping() {
  return !dag_proxy_.HasNext() || stopping_;
}

}  // namespace act
}  // namespace graphlearn
