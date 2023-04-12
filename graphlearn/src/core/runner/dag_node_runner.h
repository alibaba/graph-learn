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

#ifndef GRAPHLEARN_CORE_RUNNER_DAG_NODE_RUNNER_H_
#define GRAPHLEARN_CORE_RUNNER_DAG_NODE_RUNNER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include "core/dag/dag_node.h"
#include "core/dag/tape.h"
#include "core/operator/op_factory.h"
#include "platform/env.h"

namespace graphlearn {

class RequestFactory;

class DagNodeRunner {
public:
  explicit DagNodeRunner(Env* env);
  ~DagNodeRunner() = default;

  void Run(const DagNode* node, Tape* tape);

private:
  bool BuildInput(
    const DagNode* node, Tape* tape,
    TensorMap* tensors);

  std::unique_ptr<OpResponse> RunOp(
    const DagNode* node,
    const TensorMap& tensors);

  std::unique_ptr<OpRequest> MakeOpRequest(
    const std::string& op_name,
    const Tensor::Map& params,
    const Tensor::Map& tensors,
    const SparseTensor::Map& sparse_tensors);

private:
  Env*            env_;
  RequestFactory* req_factory_;
  op::OpFactory*  op_factory_;
};

}  // namespace graphlearn

#endif  // GRAPHLEARN_CORE_RUNNER_DAG_NODE_RUNNER_H_
