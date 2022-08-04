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

#include "actor/operator/node_getter.act.h"

#include "actor/dag/dag_actor_manager.h"
#include "core/dag/dag_node.h"
#include "include/tensor.h"

namespace graphlearn {
namespace act {

NodeGetterActor::NodeGetterActor(hiactor::actor_base* exec_ctx,
                                 const hiactor::byte_t* addr)
    : BaseOperatorActor(exec_ctx, addr) {
  set_max_concurrency(1);  // stateful
  SetOp("GetNodes");
  auto& mgr = DagActorManager::GetInstance();
  const auto* actor_params = reinterpret_cast<const OpActorParams*>(
      mgr.GetActorParams(actor_id()));
  auto &tm = actor_params->node->Params();
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

NodeGetterActor::~NodeGetterActor() {
  delete generator_;
}

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
    static_cast<int32_t>(hiactor::local_shard_id()))->GetNoder(node_type_);
  auto* id_array = noder->GetLocalStorage()->GetIds().data();

  auto begin = id_array + offset;
  tm.tensors_[kNodeIds].AddInt64(begin, begin + length);
  return seastar::make_ready_future<TensorMap>(std::move(tm));
}

}  // namespace act
}  // namespace graphlearn
