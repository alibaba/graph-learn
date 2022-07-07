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

#ifndef GRAPHLEARN_CORE_OPERATOR_SUBGRAPH_SUBGRAPH_SAMPLER_H_
#define GRAPHLEARN_CORE_OPERATOR_SUBGRAPH_SUBGRAPH_SAMPLER_H_

#include <set>
#include <string>

#include "common/base/errors.h"
#include "common/base/macros.h"
#include "core/operator/operator.h"
#include "core/operator/op_registry.h"
#include "include/subgraph_request.h"

namespace graphlearn {
namespace op {

class SubGraphSampler : public RemoteOperator {
public:
  virtual ~SubGraphSampler() {}

  Status Process(const OpRequest* req,
                 OpResponse* res) override {
    const SubGraphRequest* request =
      static_cast<const SubGraphRequest*>(req);
    SubGraphResponse* response =
      static_cast<SubGraphResponse*>(res);

    std::set<int64_t> nodes_set;
    Status s = SampleSeed(&nodes_set,
                          graph_store_,
                          request->SeedType(),
                          request->BatchSize(),
                          request->Epoch());
    RETURN_IF_NOT_OK(s);
    s = InduceSubGraph(nodes_set, request, response);
    return s;
  }

  Status Call(int32_t remote_id,
              const OpRequest* req,
              OpResponse* res) override {
    return Process(req, res);
  }

public:
  virtual Status SampleSeed(std::set<int64_t>* nodes_set,
                            GraphStore* graph_store,
                            const std::string& type,
                            int32_t batch_size,
                            int32_t epoch) = 0;
  virtual Status InduceSubGraph(
      const std::set<int64_t>& nodes_set,
      const SubGraphRequest* request,
      SubGraphResponse* response);
};

}  // namespace op
}  // namespace graphlearn

#endif  //GRAPHLEARN_CORE_OPERATOR_SUBGRAPH_SUBGRAPH_SAMPLER_H_
