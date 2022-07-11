/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "core/execution/dag_node.h"

#define LOG_FATAL_IF_PARAM_NOT_EXIST(param_map, name) \
if (param_map.find(name) == param_map.end()) {         \
  LOG(FATAL) << "Param " << name << " doesn't exist";  \
}                                                      \

namespace dgs {
namespace execution {

DagNode::DagNode(OpFactory* op_factory, const PlanNodeRep* plan_node_rep)
  : id_(plan_node_rep->id()),
    kind_(plan_node_rep->kind()),
    object_type_(plan_node_rep->type()) {
  if (plan_node_rep->params() != nullptr) {
    for (auto param : *plan_node_rep->params()) {
      param_map_.emplace(param->key()->str(), param->value());
    }
  }

  if (kind_ == PlanNode::Kind_SOURCE) {
    // Source DagNode is a dummy node, which will not run with an operator.
    op_ = nullptr;
  } else {
    std::string op_name;
    if (kind_ == PlanNode::Kind_VERTEX_SAMPLER &&
        object_type_ == PlanNode::ObjectType_VERTEX) {
      LOG_FATAL_IF_PARAM_NOT_EXIST(param_map_, "vtype");
      LOG_FATAL_IF_PARAM_NOT_EXIST(param_map_, "versions");
      op_name = "LookupVertexOp";
    } else {
      LOG_FATAL_IF_PARAM_NOT_EXIST(param_map_, "etype");
      LOG_FATAL_IF_PARAM_NOT_EXIST(param_map_, "strategy");
      LOG_FATAL_IF_PARAM_NOT_EXIST(param_map_, "fanout");
      op_name = "LookupEdgeOp";
    }

    op_ = op_factory->Create(op_name, id_, param_map_);
    if (op_ == nullptr) {
      LOG(ERROR) << "Required Opearator " << op_name <<  " not installed.";
    }
  }
}

seastar::future<Op::Records> DagNode::Execute(
    std::vector<Tensor>&& inputs,
    QueryResponseBuilder* builder,
    const storage::SampleStore* store) const {
  return op_->Process(store, std::move(inputs), builder);
}

}  // namespace execution
}  // namespace dgs
