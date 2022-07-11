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

#ifndef DGS_CORE_EXECUTION_OP_H_
#define DGS_CORE_EXECUTION_OP_H_

#include "common/log.h"
#include "common/typedefs.h"
#include "core/execution/tensor.h"
#include "core/storage/sample_store.h"
#include "service/request/query_response.h"

namespace dgs {
namespace execution {

// Stateful op
class Op {
public:
  using Params = std::unordered_map<std::string, OpParamType>;
  using Records = std::vector<io::Record>;

public:
  Op() = default;
  virtual ~Op() = default;

  virtual seastar::future<Records> Process(
    const storage::SampleStore* store,
    std::vector<Tensor>&& inputs,
    QueryResponseBuilder* builder) = 0;

  void SetParams(OperatorId opid, const Params& params) {
    id_ = opid;
    auto iter = params.find("vtype");
    if (iter != params.end()) {
      vtype_ = iter->second;
    } else {
      LOG(ERROR) << "Wrong params for Op " << opid;
    }
  }

  OperatorId id() const { return id_; }

  void ResetState() {
    unique_keys_.clear();
  }

protected:
  OperatorId id_;
  VertexType vtype_;

  // Remove the duplicated lookup.
  std::unordered_set<VertexId> unique_keys_;
};

}  // namespace execution
}  // namespace dgs

#endif  // DGS_CORE_EXECUTION_OP_H_
