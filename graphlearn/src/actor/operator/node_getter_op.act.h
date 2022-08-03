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

#ifndef GRAPHLEARN_ACTOR_OPERATOR_NODE_GETTER_OP_ACT_H_
#define GRAPHLEARN_ACTOR_OPERATOR_NODE_GETTER_OP_ACT_H_

#include <string>
#include "actor/operator/batch_generator.h"
#include "actor/operator/stateful_base_op_actor.act.h"

namespace graphlearn {
namespace actor {

class ANNOTATION(actor:reference) NodeGetterActorRef
  : public StatefulBaseOperatorActorRef {
public:
  seastar::future<TensorMap> Process(TensorMap&& Tensor) override;

  // Constructor
  ACTOR_ITFC_CTOR(NodeGetterActorRef);
  // Destructor
  ACTOR_ITFC_DTOR(NodeGetterActorRef);
};

class ANNOTATION(actor:implement) NodeGetterActor 
  : public StatefulBaseOperatorActor {
public:
  seastar::future<TensorMap> Process(TensorMap&& tensors) override;

  // Constructor
  ACTOR_IMPL_CTOR(NodeGetterActor);
  // Destructor
  ACTOR_IMPL_DTOR(NodeGetterActor);
  // Do work
  ACTOR_DO_WORK() override;

private:
  seastar::future<TensorMap> DelegateFetchData(TensorMap&& tensors);

private:
  NodeBatchGenerator* generator_;
  std::string         node_type_;
  std::string         strategy_;
  int32_t             node_from_;
  int32_t             batch_size_;
  int32_t             epoch_;
};

}  // namespace actor
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_OPERATOR_NODE_GETTER_OP_ACT_H_
