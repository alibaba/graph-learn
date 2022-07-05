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

#ifndef GRAPHLEARN_CORE_DAG_DAG_EDGE_H_
#define GRAPHLEARN_CORE_DAG_DAG_EDGE_H_

#include <memory>
#include <string>
#include "core/dag/tape.h"
#include "generated/proto/dag.pb.h"

namespace graphlearn {

class DagNode;

class BaseDagEdge {
public:
  /// A dag is made up of multi DagNodes, and each DagNode is related with
  /// an operation. When and how to run a DagNode is determined by the
  /// scheduler. The scheduling message is passed through DagEdge. The
  /// ScheduleFunction defines how an DagEdge behaves when sending or
  /// receiving scheduling messages.
  ///
  /// In the memory dag edges, maybe just forward the messages to the related
  /// nodes. In the remote dag edges, we may need some RPC calls and handle
  /// the response. More complicated operations are expected for the fused
  /// edges that will be supported later.
  using ScheduleFunction = std::function<void(DagNode*)>;

  virtual ~BaseDagEdge() {}

  int Id() const {
    return id_;
  }

  const DagNode* Src() const {
    return src_;
  }

  const DagNode* Dst() const {
    return dst_;
  }

  void SetSrc(DagNode* node) {
    src_ = node;
  }

  void SetDst(DagNode* node) {
    dst_ = node;
  }

  const std::string& SrcOutput() const {
    return src_output_;
  }

  const std::string& DstInput() const {
    return dst_input_;
  }

  virtual void Send(const ScheduleFunction& func) = 0;
  virtual void Recv(const ScheduleFunction& func) = 0;

protected:
  int      id_;
  DagNode* src_;
  DagNode* dst_;

  /// A DagEdge connects two DagNodes, that is, the upstream node generate
  /// its values and the downstream node will take the values as input. The
  /// values will be transfered through DagEdge. Usually, a DagNode will
  /// generate more than one outputs, or take more than one inputs. We need
  /// to know each output of the upstream node directs to which input of the
  /// downstream node. The next two fields are designed for this.
  std::string src_output_;
  std::string dst_input_;
};

class InMemoryDagEdge : public BaseDagEdge {
public:
  explicit InMemoryDagEdge(const DagEdgeDef& edge_def);

  void Send(const ScheduleFunction& func) override;
  void Recv(const ScheduleFunction& func) override;
};

typedef std::shared_ptr<BaseDagEdge> DagEdgePtr;
DagEdgePtr LookupOrCreateDagEdge(const DagEdgeDef& edge_def);

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_DAG_DAG_EDGE_H_
