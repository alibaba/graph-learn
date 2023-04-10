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

#ifndef GRAPHLEARN_ACTOR_OPERATOR_BASE_OP_ACT_H_
#define GRAPHLEARN_ACTOR_OPERATOR_BASE_OP_ACT_H_

#include "hiactor/core/actor-template.hh"
#include "hiactor/util/data_type.hh"

#include "actor/tensor_partitioner.h"
#include "actor/tensor_serializer.h"
#include "core/operator/operator.h"

namespace graphlearn {
namespace act {

class ANNOTATION(actor:impl) BaseOperatorActor : public hiactor::actor {
public:
  BaseOperatorActor(hiactor::actor_base* exec_ctx, const hiactor::byte_t* addr);
  ~BaseOperatorActor() override;

  virtual seastar::future<TensorMapSerializer>
  ANNOTATION(actor:method) Process(TensorMapSerializer&& tensor) = 0;

  ACTOR_DO_WORK()

protected:
  void SetOp(const std::string& op_name);
  const Tensor::Map& GetParams();

protected:
  op::Operator*      impl_;
  const Tensor::Map* params_;
};

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_OPERATOR_BASE_OP_ACT_H_
