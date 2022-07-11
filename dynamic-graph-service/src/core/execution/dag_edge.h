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

#ifndef DGS_CORE_EXECUTION_DAG_EDGE_H_
#define DGS_CORE_EXECUTION_DAG_EDGE_H_

#include "common/typedefs.h"

namespace dgs {
namespace execution {

class Dag;
class DagNode;

class DagEdge {
public:
  DagEdge(DagNode* src, DagNode* dst,
          FieldIndex src_output, FieldIndex dst_input)
    : src_(src),
      dst_(dst),
      src_output_(src_output),
      dst_input_(dst_input) {}
  ~DagEdge() = default;

  const DagNode* src() const { return src_; }
  const DagNode* dst() const { return dst_; }
  FieldIndex src_output() const { return src_output_; }
  FieldIndex dst_input() const { return dst_input_; }

private:
  friend class Dag;

  DagNode*   src_;
  DagNode*   dst_;
  FieldIndex src_output_;
  FieldIndex dst_input_;
};

}  // namespace execution
}  // namespace dgs

#endif  // DGS_CORE_EXECUTION_DAG_EDGE_H_
