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

#ifndef DGS_CORE_EXECUTION_DAG_NODE_H_
#define DGS_CORE_EXECUTION_DAG_NODE_H_

#include "core/execution/dag_edge.h"
#include "core/execution/op_factory.h"
#include "generated/fbs/query_plan_generated.h"

namespace dgs {
namespace execution {

class OpResult;

// TODO(@Seventeen17): define EdgeSet Iterator
using EdgeSet = std::vector<const DagEdge*>;

class DagNode {
public:
  DagNode(OpFactory* op_factory, const PlanNodeRep* plan_node_rep);
  ~DagNode() = default;

  const EdgeSet& in_edges() const { return in_edges_; }
  const EdgeSet& out_edges() const { return out_edges_; }

  OperatorId id() const { return id_; }
  PlanNode::Kind kind() const { return kind_; }
  PlanNode::ObjectType object_type() const { return object_type_; }

  const OpParamType GetParam(const std::string& param_name) const {
    return param_map_.at(param_name);
  }
  const ParamMap& GetParamMap() const {
    return param_map_;
  }

  seastar::future<Op::Records> Execute(
    std::vector<Tensor>&& inputs,
    QueryResponseBuilder* builder,
    const storage::SampleStore* store) const;

  void Reset() const { op_->ResetState(); }

private:
  friend class Dag;

  EdgeSet in_edges_;
  EdgeSet out_edges_;

  Op*        op_;
  OperatorId id_;  // DagNode Id is always same as Operator Id

  PlanNode::Kind       kind_;  // SAMPLER or TRAVERSE
  PlanNode::ObjectType object_type_;  // Remain field: VERTEX or EDGE
  ParamMap             param_map_;
};

}  // namespace execution
}  // namespace dgs

#endif  // DGS_CORE_EXECUTION_DAG_NODE_H_
