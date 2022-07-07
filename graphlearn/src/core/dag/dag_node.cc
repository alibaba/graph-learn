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

#include "core/dag/dag_node.h"

#include "include/constants.h"

namespace graphlearn {

DagNode::DagNode(const DagNodeDef& node_def) {
  DagNodeDef* def = const_cast<DagNodeDef*>(&node_def);

  id_ = def->id();
  op_name_ = def->op_name();

  for (int32_t i = 0; i < def->params_size(); ++i) {
    TensorValue* v = def->mutable_params(i);
    DataType type = static_cast<DataType>(v->dtype());
    ADD_TENSOR(params_, v->name(), type, v->length());
    Tensor* t = &(params_[v->name()]);
    t->SwapWithProto(v);
  }

  for (int32_t i = 0; i < def->in_edges_size(); ++i) {
    const DagEdgeDef& e = def->in_edges(i);
    DagEdgePtr dag_edge = LookupOrCreateDagEdge(e);
    dag_edge->SetDst(this);
    in_edges_.push_back(dag_edge);
  }

  for (int32_t i = 0; i < def->out_edges_size(); ++i) {
    const DagEdgeDef& e = def->out_edges(i);
    DagEdgePtr dag_edge = LookupOrCreateDagEdge(e);
    dag_edge->SetSrc(this);
    out_edges_.push_back(dag_edge);
  }
}

void DagNode::Send(const ScheduleFunction& func) {
  for (auto& edge : out_edges_) {
    edge->Recv(func);
  }
}

void DagNode::Recv(const ScheduleFunction& func) {
  func(this);
}

}  // namespace graphlearn
