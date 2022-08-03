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

#include "actor/operator/node_getter_op.act.h"

#include <utility>
#include <vector>
#include "actor/operator/op_ref_factory.h"
#include "actor/params.h"
#include "core/dag/dag_node.h"
#include "include/tensor.h"

namespace graphlearn {
namespace actor {

NodeGetterActor::NodeGetterActor(brane::actor_base *exec_ctx,
    const brane::byte_t *addr, const void* params)
    : StatefulBaseOperatorActor(exec_ctx, addr, "GetNodes") {
  auto *actor_params = reinterpret_cast<const OpActorParams*>(params);
  auto *node = actor_params->node;

  auto &tm = node->Params();
  node_type_ = tm.at(kNodeType).GetString(0);
  strategy_ = tm.at(kStrategy).GetString(0);
  node_from_ = tm.at(kNodeFrom).GetInt32(0);
  batch_size_ = tm.at(kBatchSize).GetInt32(0);
  epoch_ = tm.at(kEpoch).GetInt32(0);

  if (strategy_ == "random") {
    generator_ = new RandomNodeBatchGenerator(node_type_, batch_size_);
  } else {
    generator_ = new TraverseNodeBatchGenerator(
      node_type_, batch_size_, actor_params, strategy_);
  }
}

NodeGetterActor::~NodeGetterActor() { delete generator_; }

seastar::future<TensorMap> NodeGetterActor::Process(TensorMap&& tensors) {
  bool found =
    tensors.tensors_.find(DelegateFetchFlag) != tensors.tensors_.end();
  if (__builtin_expect(found, false)) {
    return DelegateFetchData(std::move(tensors));
  }

  return generator_->NextBatch();
}

seastar::future<TensorMap>
NodeGetterActor::DelegateFetchData(TensorMap&& tensors) {
  TensorMap tm;
  auto offset = tensors.tensors_[DelegateFetchFlag].GetInt64(0);
  auto length = tensors.tensors_[DelegateFetchFlag].GetInt64(1);
  ADD_TENSOR(tm.tensors_, kNodeIds, kInt64, length);

  Noder* noder = ShardedGraphStore::Get().OnShard(
    brane::local_shard_id())->GetNoder(node_type_);
  auto *id_array = noder->GetLocalStorage()->GetIds()->data();

  auto begin = id_array + offset;
  tm.tensors_[kNodeIds].AddInt64(begin, begin + length);
  return seastar::make_ready_future<TensorMap>(std::move(tm));
}

OpRefRegistration<NodeGetterActorRef> _NodeGetterActorRef("GetNodes");

}  // namespace actor
}  // namespace graphlearn
