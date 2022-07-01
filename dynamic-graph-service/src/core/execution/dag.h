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

#ifndef DGS_CORE_EXECUTION_DAG_H_
#define DGS_CORE_EXECUTION_DAG_H_

#include <vector>

#include "core/execution/dag_node.h"

namespace dgs {
namespace execution {

// TODO(wenting.swt): move Dag out of namespace execution
class Dag {
public:
  explicit Dag(const QueryPlanRep* query_plan_rep);
  ~Dag();

  DagNode* AddNode(OpFactory* op_factory, const PlanNodeRep* plan_node_rep);
  const DagEdge* AddEdge(DagNode* src, DagNode* dst,
                         FieldIndex src_output, FieldIndex dst_input);

  size_t num_nodes() const { return num_nodes_; }
  size_t num_edges() const { return num_edges_; }
  DagNode* GetNode(OperatorId id) const { return nodes_[id]; }
  const DagEdge* GetEdge(OperatorId id) const { return edges_[id]; }

  const DagNode* root() const { return root_; }

  // TODO(wenting.swt): define a DagNodeIter to make DagNode* immutable.
  const std::vector<DagNode*>& nodes() const { return nodes_; }

  // After run the dag a round, reset the state of op in each DagNode.
  void Reset() const;

  void DebugInfo() const;

private:
  OpFactory* op_factory_;
  DagNode*   root_;

  size_t num_nodes_;
  size_t num_edges_;

  std::vector<DagNode*> nodes_;
  std::vector<const DagEdge*> edges_;
};


}  // namespace execution
}  // namespace dgs

#endif  // DGS_CORE_EXECUTION_DAG_H_
