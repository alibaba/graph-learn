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

#ifndef GRAPHLEARN_ACTOR_OPERATOR_GRAPH_OPS_ACT_H_
#define GRAPHLEARN_ACTOR_OPERATOR_GRAPH_OPS_ACT_H_

#include "actor/operator/op_def_macro.h"
#include "actor/operator/batch_generator.h"
#include "include/graph_request.h"

namespace graphlearn {
namespace act {

class ANNOTATION(actor:impl) NodeGetterActor : public BaseOperatorActor {
public:
  NodeGetterActor(hiactor::actor_base* exec_ctx, const hiactor::byte_t* addr);
  ~NodeGetterActor() override;

  seastar::future<TensorMap>
  ANNOTATION(actor:method) Process(TensorMap&& tensors) override;

  ACTOR_DO_WORK()

private:
  seastar::future<TensorMap> DelegateFetchData(TensorMap&& tensors);

private:
  NodeBatchGenerator* generator_;
  std::string         node_type_;
};

class ANNOTATION(actor:impl) EdgeGetterActor : public BaseOperatorActor {
public:
  EdgeGetterActor(hiactor::actor_base* exec_ctx, const hiactor::byte_t* addr);
  ~EdgeGetterActor() override;

  seastar::future<TensorMap>
  ANNOTATION(actor:method) Process(TensorMap&& tensors) override;

  ACTOR_DO_WORK()

private:
  seastar::future<TensorMap> DelegateFetchData(TensorMap&& tensors);

private:
  EdgeBatchGenerator* generator_;
  std::string         edge_type_;
};

DEFINE_OP_ACTOR(GetDegree, GetDegreeRequest, GetDegreeResponse)
DEFINE_OP_ACTOR(LookupNodes, LookupNodesRequest, LookupNodesResponse)
DEFINE_OP_ACTOR(LookupEdges, LookupEdgesRequest, LookupEdgesResponse)

}  // namespace act
}  // namespace graphlearn

#endif  // GRAPHLEARN_ACTOR_OPERATOR_GRAPH_OPS_ACT_H_
