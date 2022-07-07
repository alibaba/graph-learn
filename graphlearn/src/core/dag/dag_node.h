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

#ifndef GRAPHLEARN_CORE_DAG_DAG_NODE_H_
#define GRAPHLEARN_CORE_DAG_DAG_NODE_H_

#include <string>
#include <unordered_map>
#include <vector>
#include "core/dag/dag_edge.h"
#include "include/tensor.h"
#include "generated/proto/dag.pb.h"

namespace graphlearn {

class DagNode {
public:
  /// A dag is made up of multi DagNodes, and each DagNode is related with
  /// an operation. When and how to run a DagNode is determined by the
  /// scheduler. The scheduling message is passed through DagEdge. The
  /// ScheduleFunction defines how an DagNode behaves when sending or
  /// receiving scheduling messages.
  using ScheduleFunction = std::function<void(DagNode*)>;

  explicit DagNode(const DagNodeDef& node_def);
  ~DagNode() = default;

  int32_t Id() const {
    return id_;
  }

  const std::string& OpName() const {
    return op_name_;
  }

  bool IsSink() const {
    return op_name_ == "Sink";
  }

  const Tensor::Map& Params() const {
    return params_;
  }

  const std::vector<DagEdgePtr>& InEdges() const {
    return in_edges_;
  }

  const std::vector<DagEdgePtr>& OutEdges() const {
    return out_edges_;
  }

  int32_t InDegree() const {
    return in_edges_.size();
  }

  int32_t OutDegree() const {
    return out_edges_.size();
  }

  void Send(const ScheduleFunction& func);
  void Recv(const ScheduleFunction& func);

private:
  int32_t     id_;
  Tensor::Map params_;
  std::string op_name_;
  std::vector<DagEdgePtr> in_edges_;
  std::vector<DagEdgePtr> out_edges_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_DAG_DAG_NODE_H_
