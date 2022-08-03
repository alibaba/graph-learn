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

#include "actor/operator/full_sampler.act.h"

#include <utility>
#include "actor/operator/op_ref_factory.h"
#include "actor/params.h"
#include "include/sampling_request.h"

namespace graphlearn {
namespace actor {

FullSamplerActor::FullSamplerActor(brane::actor_base *exec_ctx,
    const brane::byte_t *addr, const void* params)
    : StatelessBaseOperatorActor(exec_ctx, addr, "FullSampler") {
  auto *actor_params = reinterpret_cast<const OpActorParams*>(params);
  auto &tm = actor_params->node->Params();

  edge_type_ = tm.at(kEdgeType).GetString(0);
  sampling_strategy_ = tm.at(kStrategy).GetString(0);
}

FullSamplerActor::~FullSamplerActor() {}

seastar::future<TensorMap> FullSamplerActor::Process(TensorMap&& tensors) {
  // create request
  SamplingRequest request(edge_type_, sampling_strategy_, 0);
  request.Set(std::move(tensors.tensors_));
  SamplingResponse response;
  impl_->Process(&request, &response);

  return seastar::make_ready_future<TensorMap>(std::move(response.tensors_));
}

OpRefRegistration<FullSamplerActorRef> _FullSamplerActorRef("FullSampler");

}  // namespace actor
}  // namespace graphlearn