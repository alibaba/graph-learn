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

#include "graphlearn/core/dag/dag.h"

#include "graphlearn/common/base/errors.h"
#include "graphlearn/common/threading/sync/lock.h"
#include "graphlearn/core/dag/optimizer.h"

namespace graphlearn {

Dag::Dag(const DagDef& dag_def): id_(dag_def.id()) {
  debug_ = dag_def.DebugString();
  for (int32_t i = 0; i < dag_def.nodes_size(); ++i) {
    const DagNodeDef& node_def = dag_def.nodes(i);
    DagNode* node = new DagNode(node_def);
    nodes_.emplace_back(node);

    /// In our current cases, just ONE node has no dependencies in the dag.
    /// If cases changed, or optimization supported, such as MERGE more
    /// than one dag together, multi root nodes should be supported.
    if (node->InDegree() == 0) {
      root_ = node;
    }
  }
}

void Dag::Compile(Optimizer* optimizer) {
  /// Do some optimization, such as node fusion, to increase efficiency.
  optimizer->Optimize(this);
}

DagFactory::~DagFactory() {
  for (auto& it : map_) {
    delete it.second;
  }
}

Status DagFactory::Create(const DagDef& def, Dag** dag) {
  ScopedLocker<std::mutex> _(&mtx_);
  int32_t dag_id = def.id();
  if (map_.find(dag_id) != map_.end()) {
    return error::AlreadyExists("Dag has already existed.");
  }

  *dag = new Dag(def);
  map_[dag_id] = *dag;
  return Status::OK();
}

Dag* DagFactory::Lookup(int32_t dag_id) {
  ScopedLocker<std::mutex> _(&mtx_);
  auto it = map_.find(dag_id);
  if (it != map_.end()) {
    return it->second;
  }
  return nullptr;
}

}  // namespace graphlearn
