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

#ifndef GRAPHLEARN_ACTOR_OPERATOR_FULL_SAMPLER_ACT_H_
#define GRAPHLEARN_ACTOR_OPERATOR_FULL_SAMPLER_ACT_H_

#include "actor/operator/base_op.act.h"

namespace graphlearn {
namespace act {

class ANNOTATION(actor:impl) FullSamplerActor : public BaseOperatorActor {
public:
  FullSamplerActor(hiactor::actor_base* exec_ctx, const hiactor::byte_t* addr);
  ~FullSamplerActor() override;

  seastar::future<TensorMap>
  ANNOTATION(actor:method) Process(TensorMap&& tensors) override;

  ACTOR_DO_WORK()

private:
  std::string edge_type_;
  std::string sampling_strategy_;
  int32_t     neighbor_count_;
};

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_OPERATOR_FULL_SAMPLER_ACT_H_
