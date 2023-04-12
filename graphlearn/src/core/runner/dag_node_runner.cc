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

#include "core/runner/dag_node_runner.h"

#include <string>
#include <utility>
#include "common/base/errors.h"
#include "core/runner/op_runner.h"
#include "include/op_request.h"

namespace graphlearn {

DagNodeRunner::DagNodeRunner(Env* env)
    : env_(env) {
  op_factory_ = op::OpFactory::GetInstance();
  req_factory_ = RequestFactory::GetInstance();
}

void DagNodeRunner::Run(const DagNode* node, Tape* tape) {
  if (node->IsSink()) {
    tape->SetReady();
    LOG(INFO) << "Runner reaches sink node, and mark the tape ready.";
    return;
  }

  TensorMap tensors;
  if (!BuildInput(node, tape, &tensors)) {
    tape->Fake();
    LOG(ERROR) << "Runner occurs error, and fake the tape.";
    return;
  }

  auto res = RunOp(node, tensors);

  if (res == nullptr) {
    tape->Fake();
  } else {
    tape->Record(node->Id(), res);
  }
}

bool DagNodeRunner::BuildInput(
    const DagNode* node, Tape* tape,
    TensorMap* tensors) {
    for (auto& edge : node->InEdges()) {
    auto src_node = edge->Src();
    auto& record = tape->Retrieval(src_node->Id());
    auto ret = record.Find(edge->SrcOutput());
    auto values = std::get<0>(ret);
    auto segments = std::get<1>(ret);
    if (!tensors->Add(edge->DstInput(), values, segments)) {
      LOG(ERROR) << "DagEdge has no src node: " << src_node->Id();
      return false;
    }
  }
  return true;
}

std::unique_ptr<OpResponse> DagNodeRunner::RunOp(
    const DagNode* node,
    const TensorMap& tensors) {
  auto op_name = node->OpName();
  op::Operator* op = op_factory_->Create(op_name);
  if (op == nullptr) {
    LOG(ERROR) << "Invalid dag node: " << op_name;
    return nullptr;
  }

  auto req = MakeOpRequest(
    op_name, node->Params(), tensors.tensors_, tensors.sparse_tensors_);
  auto res = std::unique_ptr<OpResponse>(
    req_factory_->NewResponse(op_name));
  std::unique_ptr<OpRunner> runner = GetOpRunner(env_, op);
  Status s = runner->Run(req.get(), res.get());

  if (s.ok()) {
    return res;
  } else if (error::IsOutOfRange(s)) {
    LOG(INFO) << "Finish an epoch: " << op_name;
  } else {
    LOG(ERROR) << "Run dag node failed: " << op_name
               << ", details: " << s.ToString();
  }
  return nullptr;
}

std::unique_ptr<OpRequest> DagNodeRunner::MakeOpRequest(
    const std::string& op_name,
    const Tensor::Map& params,
    const Tensor::Map& tensors,
    const SparseTensor::Map& sparse_tensors) {
  OpRequest* req = req_factory_->NewRequest(op_name);
  req->Init(params);
  req->Set(tensors, sparse_tensors);
  return std::unique_ptr<OpRequest>(req);
}

}  // namespace graphlearn
