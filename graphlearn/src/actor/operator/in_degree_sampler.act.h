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

#ifndef GRAPHLEARN_ACTOR_OPERATOR_IN_DEGREE_SAMPLER_ACT_H_
#define GRAPHLEARN_ACTOR_OPERATOR_IN_DEGREE_SAMPLER_ACT_H_

#include <string>
#include "actor/operator/stateless_base_op_actor.act.h"

namespace graphlearn {
namespace actor {

class ANNOTATION(actor:reference) InDegreeSamplerActorRef
  : public StatelessBaseOperatorActorRef {
public:
  seastar::future<TensorMap> Process(TensorMap&& Tensor) override;

  // Constructor
  ACTOR_ITFC_CTOR(InDegreeSamplerActorRef);
  // Destructor
  ACTOR_ITFC_DTOR(InDegreeSamplerActorRef);
};

class ANNOTATION(actor:implement) InDegreeSamplerActor
  : public StatelessBaseOperatorActor {
public:
  seastar::future<TensorMap> Process(TensorMap&& tensors) override;
  // Constructor
  ACTOR_IMPL_CTOR(InDegreeSamplerActor);
  // Destructor
  ACTOR_IMPL_DTOR(InDegreeSamplerActor);
  // Do work
  ACTOR_DO_WORK() override;
private:
  std::string edge_type_;
  std::string sampling_strategy_;
  int32_t neighbor_count_;
};

}  // namespace actor
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_OPERATOR_IN_DEGREE_SAMPLER_ACT_H_
