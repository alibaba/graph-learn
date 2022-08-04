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

#include "include/sampling_request.h"

namespace graphlearn {
namespace act {

FullSamplerActor::FullSamplerActor(hiactor::actor_base* exec_ctx,
                                   const hiactor::byte_t* addr)
    : BaseOperatorActor(exec_ctx, addr) {
  set_max_concurrency(UINT32_MAX);  // stateless
  SetOp("FullSampler");
  auto& tm = GetParams();
  edge_type_ = tm.at(kEdgeType).GetString(0);
  sampling_strategy_ = tm.at(kStrategy).GetString(0);
}

FullSamplerActor::~FullSamplerActor() = default;

seastar::future<TensorMap> FullSamplerActor::Process(TensorMap&& tensors) {
  // create request
  SamplingRequest request(edge_type_, sampling_strategy_, 0);
  request.Set(tensors.tensors_);
  SamplingResponse response;
  impl_->Process(&request, &response);

  return seastar::make_ready_future<TensorMap>(std::move(response.tensors_));
}

}  // namespace act
}  // namespace graphlearn
