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

#include "actor/operator/edge_lookuper.act.h"

#include <utility>
#include "actor/operator/op_ref_factory.h"
#include "actor/params.h"
#include "include/graph_request.h"

namespace graphlearn {
namespace actor {

EdgeLookuperActor::EdgeLookuperActor(
    brane::actor_base *exec_ctx,
    const brane::byte_t *addr,
    const void* params)
    : StatelessBaseOperatorActor(exec_ctx, addr, "LookupEdges") {
  auto *actor_params = reinterpret_cast<const OpActorParams*>(params);
  auto &tm = actor_params->node->Params();
  edge_type_ = tm.at("et").GetString(0);
}

EdgeLookuperActor::~EdgeLookuperActor() {
}

seastar::future<TensorMap> EdgeLookuperActor::Process(
    TensorMap&& tensors) {
  LookupEdgesRequest request(edge_type_);
  request.Set(std::move(tensors.tensors_));
  LookupEdgesResponse response;
  impl_->Process(&request, &response);

  return seastar::make_ready_future<TensorMap>(
    std::move(response.tensors_));
}

OpRefRegistration<EdgeLookuperActorRef> _EdgeLookuperActorRef("LookupEdges");

}  // namespace actor
}  // namespace graphlearn
