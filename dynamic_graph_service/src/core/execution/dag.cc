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

#include "core/execution/dag.h"

#include "common/log.h"

namespace dgs {
namespace execution {

Dag::Dag(const QueryPlanRep* query_plan_rep)
    : op_factory_(OpFactory::GetInstance()) {
  auto plan_nodes = query_plan_rep->plan_nodes();
  for (size_t i = 0; i < plan_nodes->size(); ++i) {
    auto plan_node = plan_nodes->Get(i);
    AddNode(op_factory_, plan_node);
  }

  for (size_t i = 0; i < plan_nodes->size(); ++i) {
    auto plan_node = plan_nodes->Get(i);
    auto links = plan_node->links();
    for (size_t j = 0; j < links->size(); ++j) {
      auto child_link = links->Get(j);
      AddEdge(GetNode(i), GetNode(child_link->node()),
              child_link->src_output(), child_link->dst_input());
    }
  }

  num_nodes_ = nodes_.size();
  num_edges_ = edges_.size();

  if (num_nodes_ > 0) {
    root_ = nodes_[0];
  } else {
    LOG(INFO) << "The Query is empty.";
  }
}

Dag::~Dag() {
  for (auto node : nodes_) {
    delete node;
  }
  for (auto edge : edges_) {
    delete edge;
  }
  nodes_.clear();
  edges_.clear();
}

DagNode* Dag::AddNode(OpFactory* op_factory, const PlanNodeRep* plan_node_rep) {
  auto node = new DagNode(op_factory, plan_node_rep);
  nodes_.push_back(node);
  return node;
}

const DagEdge* Dag::AddEdge(DagNode* src_node, DagNode* dst_node,
                            FieldIndex src_output, FieldIndex dst_input) {
  const DagEdge* edge = new DagEdge(src_node, dst_node, src_output, dst_input);
  src_node->out_edges_.push_back(edge);
  dst_node->in_edges_.push_back(edge);
  edges_.push_back(edge);
  return edge;
}

void Dag::Reset() const {
  for (auto node : nodes_) {
    if (node->id() != 0) {
      node->Reset();
    }
  }
}

void Dag::DebugInfo() const {
  LOG(INFO) << "Dag #nodes=" << num_nodes_
            << ", #edges=" << num_edges_;

  for (auto node : nodes_) {
    LOG(INFO) << "-- \n Node " << node->id()
              << ": kind=" << node->kind()
              << ", object_type=" << node->object_type()
              << ", \n ---- params are:";
    for (auto& iter : node->GetParamMap()) {
      LOG(INFO) << "------ key=" << iter.first
                << ", value=" << iter.second;
    }
    LOG(INFO) << "---- In edges are:";
    for (auto& edge : node->in_edges()) {
      LOG(INFO) << "------ src=" << edge->src()->id()
                <<", dst=" << edge->dst()->id()
                <<", src_output=" << edge->src_output()
                << ", dst_input=" << edge->dst_input();
    }
    LOG(INFO) << "---- Out edges are:";
    for (auto& edge : node->out_edges()) {
      LOG(INFO) << "------ src=" << edge->src()->id()
                << ", dst=" << edge->dst()->id()
                << ", src_output=" << edge->src_output()
                << ", dst_input={}" << edge->dst_input();
    }
  }
}

}  // namespace execution
}  // namespace dgs
