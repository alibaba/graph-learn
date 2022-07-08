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

#ifndef DGS_CORE_EXECUTION_QUERY_EXECUTOR_H_
#define DGS_CORE_EXECUTION_QUERY_EXECUTOR_H_

#include "seastar/core/when_all.hh"

#include "common/log.h"
#include "core/execution/tensor.h"
#include "core/storage/sample_store.h"
#include "service/request/query_response.h"

namespace dgs {
namespace execution {

class QueryExecutor {
public:
  // Init one instance of QueryExecutor for each req_id.
  // TODO(@Seventeen17): limit the count of concurrency executors.
  // The QueryExecutor maintain the state between one turn of query run.
  explicit QueryExecutor(const Dag* dag);
  QueryExecutor(QueryExecutor&& other) noexcept;
  ~QueryExecutor() = default;

  // Execute the query with request vid, return QueryResponse.
  seastar::future<QueryResponse> Execute(
      VertexId req_vid, const storage::SampleStore* store);

private:
  // Run the current Node and schedule the downstream nodes.
  seastar::future<> RunNode(const DagNode* node,
                            const storage::SampleStore* store);

  // Execute the logic of Node.
  seastar::future<> ExecuteLogic(const DagNode* node,
                                 std::vector<Tensor>&& inputs,
                                 const storage::SampleStore* store);

private:
  const Dag*            dag_;
  QueryResponseBuilder  res_builder_;
  std::vector<uint32_t> refs_;
  std::vector<std::unordered_map<FieldIndex, Tensor>> shared_tensors_;
};

}  // namespace execution
}  // namespace dgs

#endif  // DGS_CORE_EXECUTION_QUERY_EXECUTOR_H_
