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

#include "core/execution/query_executor.h"

#include "common/actor_wrapper.h"
#include "core/execution/dag.h"

namespace dgs {
namespace execution {

QueryExecutor::QueryExecutor(const Dag* dag)
  : dag_(dag),
    res_builder_(),
    refs_(dag->num_nodes()),
    shared_tensors_(dag->num_nodes()) {
  for (auto node : dag->nodes()) {
    refs_[node->id()] = node->in_edges().size();
  }
}

QueryExecutor::QueryExecutor(QueryExecutor&& other) noexcept
  : dag_(other.dag_),
    res_builder_(),
    refs_(std::move(other.refs_)),
    shared_tensors_(std::move(other.shared_tensors_)) {
  other.dag_ = nullptr;
}

seastar::future<QueryResponse>
QueryExecutor::Execute(VertexId req_id, const storage::SampleStore* store) {
  std::vector<VertexId> input(1, req_id);
  shared_tensors_[0].emplace(0, Tensor(std::move(input)));

  return RunNode(dag_->root(), store).then([this] () {
    res_builder_.Finish();
    auto* data = const_cast<char*>(
        reinterpret_cast<const char*>(res_builder_.BufPointer()));
    auto size = res_builder_.BufSize();
    auto buf = act::BytesBuffer(data, size,
        seastar::make_object_deleter(std::move(res_builder_)));
    QueryResponse res(std::move(buf));
    return std::move(res);
  });
}

seastar::future<>
QueryExecutor::RunNode(const DagNode* node, const storage::SampleStore* store) {
  if (!node) {
    LOG(ERROR) << "QueryExecutor run DagNode with nullptr.";
  }

  std::vector<Tensor> inputs;
  if (node->id() == 0) {
    auto iter = shared_tensors_[0].find(0);
    // We guarantee the tensor exists.
    inputs.emplace_back(iter->second.Share());
    // root use shared_tensors_[0] as both input and output tensor;
  } else {
    for (const DagEdge* edge : node->in_edges()) {
      auto& src_outputs = shared_tensors_[edge->src()->id()];
      auto iter = src_outputs.find(edge->src_output());
      if (iter == src_outputs.end()) {
        LOG(ERROR) << "Internal Error: DagEdge in query plan is wrong"
                   << ", src_output: " << edge->src_output()
                   << ", dst_input: " << edge->dst_input()
                   << ", src node: " << edge->src()->id()
                   << ", dst node: " << edge->dst()->id();
      }
      inputs.emplace_back(iter->second.Share());
    }
  }

  return ExecuteLogic(node, std::move(inputs), store).then([this, node, store] {
    std::vector<seastar::future<>> futs;
    for (auto& e : node->out_edges()) {
      auto dst = e->dst();
      if (--refs_[dst->id()] == 0) {
        futs.emplace_back(RunNode(dst, store));
      } else {
        futs.emplace_back(seastar::make_ready_future<>());
      }
    }
    return seastar::when_all_succeed(futs.begin(), futs.end()).then([] () {
      return seastar::make_ready_future<>();
    });
  });
}

seastar::future<> QueryExecutor::ExecuteLogic(
  const DagNode* node,
  std::vector<Tensor>&& inputs,
  const storage::SampleStore* store) {
#ifdef DGS_DEBUG
  LOG(INFO) << "ExecuteLogic for DagNode " << node->id();
#endif
  if (node->id() == 0) {
    /// id of root node is always 0, which is not a real operator and is only
    /// used for sharing request id between it's downstream operators.
    /// Input and output for root are same.
    return seastar::make_ready_future<>();
  }
  return node->Execute(std::move(inputs), &res_builder_, store).then(
    [this, node] (auto&& output) {
      for (auto edge : node->out_edges()) {
        auto& tensors = shared_tensors_[node->id()];
        if (tensors.find(edge->src_output()) == tensors.end()) {
          shared_tensors_[node->id()].emplace(
            edge->src_output(), Tensor(output, edge->src_output()));
        }
      }
    });
}

}  // namespace execution
}  // namespace dgs
