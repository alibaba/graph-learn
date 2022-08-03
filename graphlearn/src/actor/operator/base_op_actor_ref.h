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

#ifndef GRAPHLEARN_ACTOR_OPERATOR_BASE_OP_ACTOR_REF_ACT_H_
#define GRAPHLEARN_ACTOR_OPERATOR_BASE_OP_ACTOR_REF_ACT_H_

#include "brane/actor/actor_implementation.hh"
#include "brane/actor/reference_base.hh"
#include "brane/util/common-utils.hh"
#include "actor/tensor_map.h"
#include "core/operator/operator.h"

namespace graphlearn {
namespace actor {

class BaseOperatorActorRef : public brane::reference_base {
public:
  BaseOperatorActorRef() : brane::reference_base() {}
  virtual ~BaseOperatorActorRef() {}
  virtual seastar::future<TensorMap> Process(TensorMap&& tensor) = 0;
};

}  // namespace actor
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_OPERATOR_BASE_OP_ACTOR_REF_ACT_H_