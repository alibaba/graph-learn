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

#include "actor/operator/edge_getter.act.h"

#include <utility>
#include "actor/operator/op_ref_factory.h"
#include "actor/params.h"
#include "include/graph_request.h"
#include "include/tensor.h"

namespace graphlearn {
namespace actor {

EdgeGetterActor::EdgeGetterActor(
    brane::actor_base *exec_ctx,
    const brane::byte_t *addr,
    const void* params)
    : StatefulBaseOperatorActor(exec_ctx, addr, "GetEdges") {
  auto *actor_params = reinterpret_cast<const OpActorParams*>(params);
  auto &tm = actor_params->node->Params();

  edge_type_ = tm.at(kEdgeType).GetString(0);
  strategy_ = tm.at(kStrategy).GetString(0);
  batch_size_ = tm.at(kBatchSize).GetInt32(0);
  epoch_ = tm.at(kEpoch).GetInt32(0);

  if (strategy_ == "random") {
    generator_ = new RandomEdgeBatchGenerator(edge_type_, batch_size_);
  } else {
    generator_ = new TraverseEdgeBatchGenerator(
      edge_type_, batch_size_, actor_params, strategy_);
  }
}

EdgeGetterActor::~EdgeGetterActor() { delete generator_; }

seastar::future<TensorMap> EdgeGetterActor::Process(
    TensorMap&& tensors) {
  bool found =
    tensors.tensors_.find(DelegateFetchFlag) != tensors.tensors_.end();
  if (__builtin_expect(found, false)) {
    return DelegateFetchData(std::move(tensors));
  }
  return generator_->NextBatch();
}

seastar::future<TensorMap>
EdgeGetterActor::DelegateFetchData(TensorMap&& tensors) {
  auto offset = tensors.tensors_[DelegateFetchFlag].GetInt64(0);
  auto length = tensors.tensors_[DelegateFetchFlag].GetInt64(1);
  auto* edge_store = ShardedGraphStore::Get().OnShard(
    brane::local_shard_id())->GetGraph(edge_type_)->GetLocalStorage();

  TensorMap tm;
  ADD_TENSOR(tm.tensors_, kEdgeIds, kInt64, length);
  ADD_TENSOR(tm.tensors_, kSrcIds, kInt64, length);
  ADD_TENSOR(tm.tensors_, kDstIds, kInt64, length);
  for (io::IdType id = offset; id < offset + length; id++) {
    tm.tensors_[kEdgeIds].AddInt64(id);
    tm.tensors_[kSrcIds].AddInt64(edge_store->GetSrcId(id));
    tm.tensors_[kDstIds].AddInt64(edge_store->GetDstId(id));
  }
  return seastar::make_ready_future<TensorMap>(std::move(tm));
}

OpRefRegistration<EdgeGetterActorRef> _EdgeGetterActorRef("GetEdges");

}  // namespace actor
}  // namespace graphlearn
